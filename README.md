# Streamlit Wishes App（七夕量子曼荼羅）

このプロジェクトは、QA4U3のGroup10+47の作品「七夕量子曼荼羅」です。  
Streamlit を用いたアプリケーションで、ユーザーが自分の願い事を入力すると、それを Google スプレッドシートに記録し、既存の願い事と最適な配置順を計算してビジュアルに表示します。さらに、p5.js を利用して、願い事の配置順、  
各種言語の翻訳結果、国旗絵文字をキャンバス上に描画します。

## プロジェクト構成

```
streamlit-wishes-app-cloud
├── src
│   ├── app3.py               # 主な Streamlit アプリのエントリーポイント
│   ├── google_sheets.py      # Google Sheets API との連携（読み書き）を行うモジュール
│   ├── config
│   │   └── settings.py       # 各種設定（APIキー、スプレッドシートID、QUBOパラメータなど）
│   └── data                 # 国情報やその他のデータを格納するフォルダ
│       ├── country_flags.csv # 国旗絵文字や国情報のデータ
│       └── その他のデータファイル
├── requirements.txt          # 必要な Python パッケージ一覧
└── README.md                 # 本ドキュメント
```

## セットアップ方法

1. **リポジトリのクローン**

   ```
   git clone <repository-url>
   cd streamlit-wishes-app-cloud
   ```

2. **必要なパッケージのインストール**

   Python がインストール済みであることを確認し、以下のコマンドを実行してください。

   ```
   pip install -r requirements.txt
   ```

3. **Google Sheets の設定**

   - [Google Sheets API Quickstart](https://developers.google.com/sheets/api/quickstart/python) を参照して、Google Cloud 上でプロジェクトを設定し、サービスアカウントキー（JSON）を取得してください。
   - 取得した認証情報を、`google_sheets.py` で指定された場所に配置するか、Streamlit Cloud の Secrets を利用してください。
   - `google_sheets.py` 内で、`spreadsheet_id` や `sheet_name` などの設定を適宜修正してください。

4. **データフォルダの確認**

   - 国情報などの CSV ファイルは、`src/data` フォルダ内に配置してください。  
     例：`src/data/country_flags.csv`

5. **アプリの起動**

   以下のコマンドを実行して、アプリを起動します。

   ```
   streamlit run src/app3.py
   ```

## アプリの機能

- **ユーザー入力**  
  ユーザーは年代、性別、国（日本の場合は都道府県）と願い事を入力できます。

- **Google Sheets との連携**  
  入力された願い事は `google_sheets.py` の機能を通じて Google スプレッドシートに書き込まれ、  
  既存の願い事はスプレッドシートからランダムに読み込まれます。

- **願い事の最適配置**  
  願い事データは翻訳、ベクトル化、QUBO 最適化などの手順を経て、  
  最適な配置順が計算され、p5.js を利用してキャンバス上に国旗や翻訳結果と共に表示されます。

## 注意事項

- Google Sheets への認証情報はセキュリティ上の理由から公開しないでください。  
  Streamlit Cloud の Secret 管理機能や環境変数を利用してください。
- アプリの動作に必要な各パッケージは `requirements.txt` に記載しています。  
  各パッケージのバージョンは実際の環境に合わせ調整してください。

## 貢献方法

改善点やバグ修正の提案があれば、プルリクエストまたはイシューを作成してください。  