const sharp = require('sharp');
const fs = require('fs');
const path = require('path');
const tf = require('@tensorflow/tfjs-node');

// ディレクトリ設定
const inputDir = './input_images';
const outputDir = './output_images';

// 実行する処理のリスト
const operations = ['noise'];



// 各処理の設定
const brightnessAdjustment = 0.2; // 明るさ調整
const shiftXRatio = 0.1; // シフトの横方向割合
const shiftYRatio = -0.1; // シフトの縦方向割合
const hideRegion = { x: 50, y: 50, width: 200, height: 150 }; // ノイズで隠す範囲
const hueShiftValue = 90; // 色相シフト値
const flipMode = 'horizontal'; // 'horizontal' または 'vertical'
const cropArea = { x: 0.25, y: 0.25, width: 0.5, height: 0.5 }; // 切り抜き範囲 (割合指定)
const noiseLevel = 0.2; // ノイズの強さ（0.0～1.0）



// データ拡張用の反転関数
const augmentations = {
  flipHorizontal: (image) => tf.image.flipLeftRight(image.expandDims(0)).squeeze(0),
  flipVertical: (image) => tf.reverse(image.expandDims(0), [1]).squeeze(0),
};

// RGBをHSVに変換
function rgbToHsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const max = Math.max(r, g, b), min = Math.min(r, g, b);
    const delta = max - min;

    let h = 0;
    if (delta !== 0) {
        if (max === r) {
            h = ((g - b) / delta) % 6;
        } else if (max === g) {
            h = (b - r) / delta + 2;
        } else {
            h = (r - g) / delta + 4;
        }
    }

    h = (h * 60 + 360) % 360;
    const s = max === 0 ? 0 : delta / max;
    const v = max;

    return [h, s, v];
}

// HSVをRGBに変換
function hsvToRgb(h, s, v) {
    const c = v * s;
    const x = c * (1 - Math.abs((h / 60) % 2 - 1));
    const m = v - c;

    let r = 0, g = 0, b = 0;
    if (h >= 0 && h < 60) {
        [r, g, b] = [c, x, 0];
    } else if (h >= 60 && h < 120) {
        [r, g, b] = [x, c, 0];
    } else if (h >= 120 && h < 180) {
        [r, g, b] = [0, c, x];
    } else if (h >= 180 && h < 240) {
        [r, g, b] = [0, x, c];
    } else if (h >= 240 && h < 300) {
        [r, g, b] = [x, 0, c];
    } else {
        [r, g, b] = [c, 0, x];
    }

    r = Math.round((r + m) * 255);
    g = Math.round((g + m) * 255);
    b = Math.round((b + m) * 255);

    return [r, g, b];
}

// 明るさ変更
async function adjustBrightness(imageBuffer, adjustment) {
    return sharp(imageBuffer)
        .modulate({ brightness: 1 + adjustment })
        .toBuffer();
}

// シフト処理
async function shiftImage(imageBuffer, shiftXRatio, shiftYRatio) {
    const { data, info } = await sharp(imageBuffer).raw().toBuffer({ resolveWithObject: true });

    const width = info.width;
    const height = info.height;
    const channels = info.channels;

    const shiftX = Math.round(shiftXRatio * width);
    const shiftY = Math.round(shiftYRatio * height);

    const outputData = Buffer.alloc(data.length, 128); // ノイズ（中間値）

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const srcX = x - shiftX;
            const srcY = y - shiftY;

            if (srcX >= 0 && srcX < width && srcY >= 0 && srcY < height) {
                const srcIndex = (srcY * width + srcX) * channels;
                const dstIndex = (y * width + x) * channels;

                for (let c = 0; c < channels; c++) {
                    outputData[dstIndex + c] = data[srcIndex + c];
                }
            }
        }
    }

    return sharp(outputData, { raw: { width, height, channels } })
        .toFormat('jpeg')
        .toBuffer();
}

// ノイズで隠す
async function hideWithNoise(imageBuffer, hideRegion) {
    const { data, info } = await sharp(imageBuffer).raw().toBuffer({ resolveWithObject: true });

    const width = info.width;
    const height = info.height;
    const channels = info.channels;

    const x = hideRegion.x;
    const y = hideRegion.y;
    const hideWidth = hideRegion.width;
    const hideHeight = hideRegion.height;

    const noiseData = Buffer.alloc(hideWidth * hideHeight * channels, 128); // ノイズ
    const outputData = Buffer.from(data);

    for (let ny = 0; ny < hideHeight; ny++) {
        for (let nx = 0; nx < hideWidth; nx++) {
            const dstIndex = ((y + ny) * width + (x + nx)) * channels;

            for (let c = 0; c < channels; c++) {
                outputData[dstIndex + c] = noiseData[(ny * hideWidth + nx) * channels + c];
            }
        }
    }

    return sharp(outputData, { raw: { width, height, channels } })
        .toFormat('jpeg')
        .toBuffer();
}

// 色相変更
async function changeHue(imageBuffer, hueShift) {
    const { data, info } = await sharp(imageBuffer).raw().toBuffer({ resolveWithObject: true });
    const outputData = Buffer.from(data);

    for (let i = 0; i < data.length; i += 3) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];

        let [h, s, v] = rgbToHsv(r, g, b);
        h = (h + hueShift) % 360;

        const [newR, newG, newB] = hsvToRgb(h, s, v);
        outputData[i] = newR;
        outputData[i + 1] = newG;
        outputData[i + 2] = newB;
    }

    return sharp(outputData, { raw: { width: info.width, height: info.height, channels: 3 } })
        .toFormat('jpeg')
        .toBuffer();
}


// ノイズを追加する関数
function addNoiseToImage(imageBuffer, noiseLevel = 0.1) {
    const imageTensor = tf.node.decodeImage(imageBuffer).div(tf.scalar(255)); // 0～1に正規化
    const noise = tf.randomNormal(imageTensor.shape, 0, noiseLevel); // ノイズ生成
    const noisyImageTensor = imageTensor.add(noise).clipByValue(0, 1); // ノイズ追加＆値を0~1に制限
  
    return tf.node.encodeJpeg(noisyImageTensor.mul(255).toInt()); // ノイズ追加後に画像に戻す
}
  

// 新しい関数：反転処理
async function applyFlip(imageBuffer, mode) {
  const imageTensor = tf.node.decodeImage(imageBuffer);
  const augmentFn = mode === 'horizontal' ? augmentations.flipHorizontal : augmentations.flipVertical;
  const flippedImageTensor = augmentFn(imageTensor.toFloat());
  const buffer = await tf.node.encodeJpeg(flippedImageTensor);
  return buffer;
}

// 切り抜き処理の関数
async function cropImage(imageBuffer, cropArea) {
    const tfImage = tf.node.decodeImage(imageBuffer);
    const [height, width] = tfImage.shape;
  
    // 割合からピクセル単位に変換
    const adjustedCropArea = {
      x: Math.round(cropArea.x * width),
      y: Math.round(cropArea.y * height),
      width: Math.round(cropArea.width * width),
      height: Math.round(cropArea.height * height),
    };
  
    // 範囲調整（画像サイズを超えないように）
    adjustedCropArea.x = Math.min(adjustedCropArea.x, width);
    adjustedCropArea.y = Math.min(adjustedCropArea.y, height);
    adjustedCropArea.width = Math.min(adjustedCropArea.width, width - adjustedCropArea.x);
    adjustedCropArea.height = Math.min(adjustedCropArea.height, height - adjustedCropArea.y);
  
    if (adjustedCropArea.width <= 0 || adjustedCropArea.height <= 0) {
      throw new Error('Invalid crop area dimensions.');
    }
  
    // 切り抜き処理
    const cropped = tfImage.slice(
      [adjustedCropArea.y, adjustedCropArea.x, 0],
      [adjustedCropArea.height, adjustedCropArea.width, 3]
    );
  
    // JPEGエンコードしてバッファに戻す
    const croppedBuffer = await tf.node.encodeJpeg(cropped);
    return croppedBuffer;
  }

// 画像をリサイズする関数
async function resizeImage(imageBuffer, width, height) {
  const imageTensor = tf.node.decodeImage(imageBuffer);
  const resizedTensor = tf.image.resizeBilinear(imageTensor, [width, height]); // リサイズ処理
  return tf.node.encodeJpeg(resizedTensor);
}


// 操作適用の関数
async function applyOperations(imageBuffer) {
  let processedBuffer = imageBuffer;

  for (const operation of operations) {
    if (operation === 'crop') {
      processedBuffer = await cropImage(processedBuffer, cropArea);
    } else if (operation === 'brightness') {
      processedBuffer = await adjustBrightness(processedBuffer, brightnessAdjustment);
    } else if (operation === 'shift') {
      processedBuffer = await shiftImage(processedBuffer, shiftXRatio, shiftYRatio);
    } else if (operation === 'hide') {
      processedBuffer = await hideWithNoise(processedBuffer, hideRegion);
    } else if (operation === 'hue') {
      processedBuffer = await changeHue(processedBuffer, hueShiftValue);
    } else if (operation === 'flip') {
      processedBuffer = await applyFlip(processedBuffer, flipMode);
    } else if (operation === 'noise') {
      processedBuffer = await addNoiseToImage(processedBuffer, noiseLevel);
    } else {
      throw new Error(`Invalid operation: ${operation}`);
    }
  }

  // 最後にリサイズ処理を追加
  processedBuffer = await resizeImage(processedBuffer, 224, 224);

  return processedBuffer;
}

  
  

// ディレクトリ内の画像を処理
fs.readdir(inputDir, async (err, files) => {
  if (err) {
    console.error('Error reading input directory:', err);
    return;
  }

  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir);
  }

  for (const file of files) {
    const inputPath = path.join(inputDir, file);
    const outputPath = path.join(outputDir, file);

    try {
      const imageBuffer = fs.readFileSync(inputPath);
      const processedBuffer = await applyOperations(imageBuffer);
      fs.writeFileSync(outputPath, processedBuffer);
      console.log(`Processed and saved: ${outputPath}`);
    } catch (error) {
      console.error(`Error processing ${file}:`, error);
    }
  }
});
