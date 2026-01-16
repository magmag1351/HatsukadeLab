import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# 日本語フォントの設定 (Windows環境向け)
plt.rcParams['font.family'] = 'MS Gothic'

# 設定
# スクリプトの配置ディレクトリを基準にパスを設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'source', 'csv', '欠陥なし', '単体')
OUTPUT_DIR = os.path.join(BASE_DIR, 'comparison_output')
BASE_FILE_NAME = '単体250差動.csv'
TARGET_COLUMN = ' 加算平均値[V]'

# 軸の範囲設定 (Noneの場合は自動設定、例: (0, 100))
X_LIM = None  # x軸の範囲
Y_LIM = (-0.02, 0.02)  # y軸の範囲

def main():
    # 出力ディレクトリの作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    base_file_path = os.path.join(DATA_DIR, BASE_FILE_NAME)

    if not os.path.exists(base_file_path):
        print(f"基準ファイルが見つかりません: {base_file_path}")
        return

    # 基準データの読み込み
    try:
        # Shift_JISで読み込み
        df_base = pd.read_csv(base_file_path, encoding='shift_jis')
    except Exception as e:
        print(f"基準ファイルの読み込みエラー: {e}")
        return

    # 比較対象のファイルを検索
    # パターン: 単体250差動_filter-*.csv
    search_pattern = os.path.join(DATA_DIR, '単体250差動_filter-*.csv')
    filter_files = glob.glob(search_pattern)

    if not filter_files:
        print(f"比較対象のファイルが見つかりません: {search_pattern}")
        return

    print(f"{len(filter_files)} 個のファイルを処理します...")

    for filter_file_path in filter_files:
        filename = os.path.basename(filter_file_path)
        
        # 比較データの読み込み
        try:
            df_filter = pd.read_csv(filter_file_path, encoding='shift_jis')
        except Exception as e:
            print(f"ファイルの読み込みエラー ({filename}): {e}")
            continue

        # プロット作成
        plt.figure(figsize=(10, 6))
        
        # 基準データのプロット (黒線)
        plt.plot(df_base[TARGET_COLUMN], label=f'基準: {BASE_FILE_NAME}', color='black', alpha=0.8)
        
        # 比較データのプロット (赤線)
        plt.plot(df_filter[TARGET_COLUMN], label=f'比較: {filename}', color='red', alpha=0.8, linestyle='--')

        plt.title(f'データ比較: {BASE_FILE_NAME} vs {filename}')
        plt.xlabel('データ点 (Index)')
        plt.ylabel(TARGET_COLUMN)
        
        # 軸制限の設定
        if X_LIM is not None:
            plt.xlim(X_LIM)
        if Y_LIM is not None:
            plt.ylim(Y_LIM)

        plt.legend()
        plt.grid(True)

        # 保存
        output_filename = f"compare_{filename.replace('.csv', '.png')}"
        save_path = os.path.join(OUTPUT_DIR, output_filename)
        plt.savefig(save_path)
        plt.close()
        
        print(f"保存しました: {save_path}")

if __name__ == "__main__":
    main()
