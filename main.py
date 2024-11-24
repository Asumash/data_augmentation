import cv2
import numpy as np
import os
import random

# データ拡張処理関数
def resize_image(image, size=(224, 224)):
    """画像のリサイズ"""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

def flip_image(image):
    """水平反転"""
    return cv2.flip(image, 1)

def flip_image_vertical(image):
    """上下反転"""
    return cv2.flip(image, 0)

def crop_image(image, crop_size, position=None):
    """画像の指定範囲を切り取り"""
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    if position:
        top, left = position
    else:
        # デフォルトは中央から切り取る
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2

    # 範囲調整
    top = max(0, min(top, h - crop_h))
    left = max(0, min(left, w - crop_w))

    return image[top:top+crop_h, left:left+crop_w]

def adjust_brightness(image, alpha, beta):
    """明るさ調整"""
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def change_hue(image, hue_shift):
    """色相の変更"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def add_noise(image, noise_level):
    """ノイズの追加"""
    noise = np.random.randint(-noise_level, noise_level, image.shape, dtype='int16')
    noisy_image = np.clip(image.astype('int16') + noise, 0, 255)
    return noisy_image.astype('uint8')

def shift_image(image, shift_x, shift_y):
    """画像の上下左右方向へのシフト（隙間はノイズで埋める）"""
    h, w = image.shape[:2]
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # 隙間をノイズで埋める
    mask = cv2.warpAffine(np.ones_like(image), M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    noise = np.random.randint(0, 256, image.shape, dtype='uint8')
    shifted[mask == 0] = noise[mask == 0]
    return shifted

def mask_with_noise(image, mask_size, position=None):
    """画像の一部をノイズで隠す"""
    h, w = image.shape[:2]
    mask_h, mask_w = mask_size

    if position:
        top, left = position
    else:
        # ランダム位置
        top = random.randint(0, h - mask_h)
        left = random.randint(0, w - mask_w)

    # 範囲調整
    top = max(0, min(top, h - mask_h))
    left = max(0, min(left, w - mask_w))

    # ランダムノイズの生成
    noise = np.random.randint(0, 256, (mask_h, mask_w, 3), dtype='uint8')

    # ノイズを画像に適用
    masked_image = image.copy()
    masked_image[top:top+mask_h, left:left+mask_w] = noise
    return masked_image

# 処理リストを定義
OPERATIONS = {
    "flipped": flip_image,
    "flipped_vertical": flip_image_vertical,
    "cropped": lambda img: crop_image(img, (112, 112), position=(50, 50)),  # 指定位置とサイズ
    "brightened": lambda img: adjust_brightness(img, 1.2, 50),
    "hue_changed": lambda img: change_hue(img, 30),
    "noisy": lambda img: add_noise(img, 30),
    "shifted": lambda img: shift_image(img, 50, 30),
    "masked": lambda img: mask_with_noise(img, (50, 50), position=(50, 50))  # 指定位置でマスク
}

# 出力先フォルダを作成
def create_output_folder(base_output_dir, operation_combination):
    """処理組み合わせごとにフォルダを作成"""
    folder_name = "_".join(operation_combination)
    output_path = os.path.join(base_output_dir, folder_name)
    os.makedirs(output_path, exist_ok=True)
    return output_path

# 指定した処理を順に適用
def apply_operations(image, operation_names):
    """指定された処理を順に適用"""
    for op_name in operation_names:
        if op_name in OPERATIONS:
            image = OPERATIONS[op_name](image)
        else:
            print(f"Operation {op_name} not found.")
    return image

# 画像処理と保存
def process_images(input_dir, output_dir, operation_combinations):
    """
    各画像に指定された処理の組み合わせを適用
    :param input_dir: 入力フォルダ
    :param output_dir: 出力フォルダ
    :param operation_combinations: 実行する処理の組み合わせリスト
    """
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        # 画像読み込み
        image = cv2.imread(file_path)
        if image is None:
            print(f"Could not read {file_name}. Skipping.")
            continue

        # リサイズ
        image = resize_image(image)

        base_name, _ = os.path.splitext(file_name)

        # 各組み合わせを適用して保存
        for operations in operation_combinations:
            processed_image = apply_operations(image, operations)

            # 保存先フォルダを処理ごとに分ける
            output_subfolder = create_output_folder(output_dir, operations)

            # 出力ファイル名に処理名を含める
            operation_suffix = "_".join(operations)
            output_file = os.path.join(output_subfolder, f"{base_name}_{operation_suffix}.png")
            cv2.imwrite(output_file, processed_image)

        print(f"Processed and saved augmented images for {file_name}")

# メイン処理
if __name__ == "__main__":
    # 入力フォルダと出力フォルダを指定
    input_directory = "input_images"  # 元画像が格納されているフォルダ
    output_directory = "output_images"  # 処理結果を保存するフォルダ

    # 処理の組み合わせリストを指定
    operation_combinations = [
        ["flipped"],                          # 水平反転
        ["flipped_vertical", "noisy"],        # 上下反転 + ノイズ
        ["cropped", "brightened", "hue_changed"],  # 切り取り + 明るさ調整 + 色相変更
        ["flipped","flipped_vertical","cropped","brightened","hue_changed","noisy","shifted","masked"] #all
    ]

    # 画像処理実行
    process_images(input_directory, output_directory, operation_combinations)
    print("All images processed.")

