import streamlit as st
import os
from deep_translator import GoogleTranslator
import pandas as pd
import datetime
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# settings.py からパラメータを読み込み
from config.settings import (
    LAMBDA1, LAMBDA2, LAMBDA3, LAMBDA4, ALPHA,
    WEIGHT_COUNTRY, WEIGHT_AGE, WEIGHT_GENDER, WEIGHT_PREF,
    NUM_READS, SCHEDULE
)

# google_sheets.py の読み書き機能をインポート
from google_sheets import write_wish, read_wishes


def vectorize_wishes(wishes):
    """
    願い事をベクトル化する関数
    """
    from sentence_transformers import SentenceTransformer
    model_path = os.path.join("models", "labse-distil")
    model = SentenceTransformer(model_path)
    wish_vectors = model.encode(wishes, normalize_embeddings=True)
    return wish_vectors


def calculate_similarity_matrix(wish_vectors):
    """
    願い事のベクトルから類似度行列を計算する関数
    """
    similarity_matrix = 1.0 - np.inner(wish_vectors, wish_vectors)  # コサイン距離
    return similarity_matrix


def encode_attributes(df):
    """
    属性（国、年代、性別、都道府県）をエンコードする関数
    """
    le_country = LabelEncoder()
    le_age = LabelEncoder()
    le_gender = LabelEncoder()
    le_pref = LabelEncoder()

    # すべて文字列型に変換する
    encoded_country = le_country.fit_transform(df["country"].astype(str)).astype(int)
    encoded_age = le_age.fit_transform(df["age"].astype(str)).astype(int)
    encoded_gender = le_gender.fit_transform(df["gender"].astype(str)).astype(int)
    encoded_pref = le_pref.fit_transform(df["prefecture"].fillna("不明").astype(str)).astype(int)

    attributes = {
        "country": encoded_country,
        "age": encoded_age,
        "gender": encoded_gender,
        "prefecture": encoded_pref
    }
    return attributes


def build_qubo_with_order(similarity_matrix, attributes, order, 
                          lambda1=LAMBDA1, 
                          lambda2=LAMBDA2, 
                          lambda3=LAMBDA3, 
                          lambda4=LAMBDA4, 
                          alpha=ALPHA,
                          weight_country=WEIGHT_COUNTRY,
                          weight_age=WEIGHT_AGE,
                          weight_gender=WEIGHT_GENDER,
                          weight_pref=WEIGHT_PREF):
    """
    配置順を考慮してQUBOを構築する関数
    """
    N = len(order)
    Q = defaultdict(float)

    # 属性の違いを計算
    attr_dist = np.zeros((N, N), dtype=int)
    for i in range(N):
        for k in range(N):
            if i != k:
                attr_dist[i, k] = (
                    weight_country * int(attributes["country"][order[i]] != attributes["country"][order[k]]) +
                    weight_age * int(attributes["age"][order[i]] != attributes["age"][order[k]]) +
                    weight_gender * int(attributes["gender"][order[i]] != attributes["gender"][order[k]]) +
                    weight_pref * int(attributes["prefecture"][order[i]] != attributes["prefecture"][order[k]])
                )

    # QUBOの構築
    wj_list = [np.exp(alpha * (N - idx - 1)) for idx in range(N - 1)]
    for idx in range(N - 1):
        i, k = order[idx], order[idx + 1]
        wj = wj_list[idx]
        Q[(i, k)] += lambda1 * wj * similarity_matrix[i, k] 
        Q[(i, k)] -= lambda2 * attr_dist[i, k]

    # 制約条件
    for i in range(N):
        for j1 in range(N):
            for j2 in range(j1 + 1, N):
                Q[((i, j1), (i, j2))] += lambda3

    for j in range(N):
        for i1 in range(N):
            for i2 in range(i1 + 1, N):
                Q[((i1, j), (i2, j))] += lambda4

    fixed_i = order[0]  # 起点の願い事を先頭に固定
    Q[((fixed_i, 0), (fixed_i, 0))] -= 10000.0

    return Q


def optimize_wish_order(Q):
    """
    QUBOを解いて最適な配置順を決定する関数
    """
    from openjij import SQASampler
    from dimod import BinaryQuadraticModel
    bqm = BinaryQuadraticModel.empty(vartype='BINARY')
    for (u, v), bias in Q.items():
        u_str = f"{u[0]}_{u[1]}" if isinstance(u, tuple) else str(u)
        v_str = f"{v[0]}_{v[1]}" if isinstance(v, tuple) else str(v)
        if u_str != v_str:
            bqm.add_interaction(u_str, v_str, bias)
        else:
            bqm.add_variable(u_str, bias)

    sampler = SQASampler()
    response = sampler.sample(bqm, num_reads=NUM_READS, schedule=SCHEDULE)

    best_sample = response.first.sample
    position_map = {}
    for key, value in best_sample.items():
        if value == 1 and '_' in key:
            i_str, j_str = key.split('_')
            i, j = int(i_str), int(j_str)
            position_map[i] = j

    return position_map


def assign_ranks(position_map, start_idx=0, total_length=None):
    """
    配置順を連番に変換する関数
    """
    if total_length is None:
        total_length = max(position_map.keys(), default=-1) + 1

    rank_map = {}
    assigned = set()

    if start_idx in position_map:
        rank_map[start_idx] = 0
        assigned.add(0)

    positions = sorted((j, i) for i, j in position_map.items())
    for j, i in positions:
        while j in assigned:
            j += 1
        rank_map[i] = j
        assigned.add(j) 

    unplaced_indices = [i for i in range(total_length) if i not in rank_map]
    start_rank = max(rank_map.values(), default=-1) + 1
    for offset, i in enumerate(unplaced_indices):
        rank_map[i] = start_rank + offset

    return rank_map


def result_to_df(df, position_map):
    """
    配置順をDataFrameに変換する関数
    """
    df_result = df.copy()
    for i, j in position_map.items():
        df_result.loc[i, "配置順"] = j
    if "配置順" in df_result.columns:
        df_result = df_result.sort_values("配置順")
        df_result["配置順"] = list(range(len(df_result)))

    return df_result


def process_wishes_with_fixed_start(user_wish, df):
    """
    ユーザーの願い事を起点にして配置順を決定する関数
    """
    try:
        with st.spinner("あなたの願い事に追加中..."):
            user_data = {
                "age": user_wish["age"],
                "gender": user_wish["gender"],
                "country": user_wish["country"],
                "prefecture": user_wish.get("prefecture", None),
                "wish_original": user_wish["wish_original"],
                "wish_ja": user_wish["wish_ja"],
                "wish_en": user_wish["wish_en"],
                "wish_zh": user_wish["wish_zh"],
                "wish_ko": user_wish["wish_ko"],
                "flag": user_wish["flag"],
            }
            df = pd.concat([pd.DataFrame([user_data]), df], ignore_index=True)

        with st.spinner("属性をエンコード中..."):
            attributes = encode_attributes(df)

        with st.spinner("願い事をベクトル化中..."):
            wishes = df["wish_original"].tolist()
            wish_vectors = vectorize_wishes(wishes)

        with st.spinner("類似度行列を計算中..."):
            similarity_matrix = calculate_similarity_matrix(wish_vectors)

        with st.spinner("配置順を初期化中..."):
            N = len(wishes)
            start_idx = 0
            remaining_indices = [i for i in range(N) if i != start_idx]
            remaining_indices.sort(key=lambda i: similarity_matrix[start_idx, i])
            ordered_indices = [start_idx] + remaining_indices

        with st.spinner("QUBOを構築中..."):
            Q = build_qubo_with_order(similarity_matrix, attributes, ordered_indices)

        with st.spinner("QUBOを最適化中..."):
            position_map = optimize_wish_order(Q)

        with st.spinner("配置順を変換中..."):
            rank_map = assign_ranks(position_map, start_idx=0, total_length=len(df))

        with st.spinner("結果を格納中..."):
            df_result = result_to_df(df, rank_map)

        st.write("宇宙のどこかにある　あなたの願い事銀河が見つかりました")
        st.write("Somewhere in the universe, your wish galaxy has been found!")
        return df_result
    
    except Exception as e:
        st.write(f"エラーが発生しました: {e}")
        return pd.DataFrame()


def translate_wish(text, target_languages):
    """願い事を指定された言語に翻訳"""
    translations = {}
    for lang in target_languages:
        try:
            translations[lang] = GoogleTranslator(source='auto', target=lang).translate(text)
        except Exception as e:
            translations[lang] = f"翻訳エラー: {e}"
    return translations


def render_p5_spiral(wish_original, ranks, flags=None, wish_en=None, wish_ja=None, wish_zh=None, wish_ko=None):
    import streamlit.components.v1 as components
    import json
    flags = flags or [""] * len(wish_original)
    wish_en = wish_en or [""] * len(wish_original)
    wish_ja = wish_ja or [""] * len(wish_original)
    wish_zh = wish_zh or [""] * len(wish_original)
    wish_ko = wish_ko or [""] * len(wish_original)
    html_code = f"""
    <html>
      <head>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.6.0/p5.min.js"></script>
        <script>
            const wish_original = {json.dumps(wish_original)};
            const flags = {json.dumps(flags)};
            const wish_en = {json.dumps(wish_en)};
            const wish_ja = {json.dumps(wish_ja)};
            const wish_zh = {json.dumps(wish_zh)};
            const wish_ko = {json.dumps(wish_ko)};
            const ranks = {json.dumps(ranks)};

            let wishes = wish_original; 
            let stars = [];
            const STAR_COUNT = 10000;
            let positions = [];
            let randomEdges = [];
            let langSelect;
            let plusButton, minusButton, fsButton;
            let scaleFactor = 1, translateX = 0, translateY = 0;
            let isDragging = false, lastMouseX, lastMouseY;

            function toggleFullscreen() {{
                let fs = fullscreen();
                fullscreen(!fs);
            }}

            function zoomIn() {{
                scaleFactor *= 1.1;
            }}

            function zoomOut() {{
                scaleFactor /= 1.1;
            }}

            function setup() {{
                createCanvas(windowWidth, windowHeight);
                textFont("Arial, sans-serif");

                for (let i = 0; i < STAR_COUNT; i++) {{
                    stars.push({{
                    x: random(-width*4, width*4),
                    y: random(-height*4, height*4),
                    size: random(1, 3),
                    twinkleSpeed: random(0.01, 0.05),
                    phase: random(TWO_PI)
                    }});
                }}

                langSelect = createSelect();
                langSelect.position(20, 20);
                langSelect.option('Original');
                langSelect.option('日本語');
                langSelect.option('English');
                langSelect.option('中文');
                langSelect.option('한국어');
                langSelect.selected('Original');
                langSelect.changed(updateLanguage);

                plusButton = createButton('+');
                plusButton.position(20, 50);
                plusButton.mousePressed(zoomIn);

                minusButton = createButton('-');
                minusButton.position(60, 50);
                minusButton.mousePressed(zoomOut);

                fsButton = createButton('⛶ Fullscreen');
                fsButton.position(100, 50);
                fsButton.mousePressed(toggleFullscreen);

                calculatePositions();
                generateRandomEdges();
            }}

            function updateLanguage() {{
                const v = langSelect.value();
                if (v === 'Original' || v === '原文') wishes = wish_original;
                else if (v === '日本語') wishes = wish_ja;
                else if (v === 'English') wishes = wish_en;
                else if (v === '中文') wishes = wish_zh;
                else if (v === '한국어') wishes = wish_ko;
            }}

            function calculatePositions() {{
                const total = wish_original.length;
                let margin = map(total, 10, 100, 100, 2000);
                margin = constrain(margin, 50, 3000);
                const maxRadius = min(width, height) / 2 + margin;
                const fixedDistance = 300;
                let rotations = map(total, 10, 100, 2, 8);
                rotations = constrain(rotations, 2, 10);

                positions = [];
                const jitterMax = radians(10);
                const maxRank = total - 1;

                for (let i = 0; i < total; i++) {{
                    let r, angle;
                    if (i === 0) {{ r = 0; angle = 0; }}
                    else if (i === 1) {{ r = fixedDistance; angle = 0; }}
                    else {{
                        const t = i - 1;
                        angle = t * (TWO_PI * rotations / (maxRank - 1));
                        angle += random(-jitterMax, jitterMax);
                        r = map(t, 1, maxRank - 1, fixedDistance + 20, maxRadius);
                    }}
                    positions.push({{ x: width/2+cos(angle)*r, y: height/2+sin(angle)*r }});
                }}
            }}

            function generateRandomEdges() {{
                randomEdges = [];
                for (let k = 0; k < positions.length; k++) {{
                    let i = floor(random(1, positions.length));
                    let j = floor(random(1, positions.length));
                    if (i !== j) randomEdges.push([i,j]);
                }}
            }}

            function wrapText(str, maxChars) {{
                let lines = [];
                for (let i = 0; i < str.length; i += maxChars) {{
                    lines.push(str.substring(i, i + maxChars));
                }}
                return lines;
            }}

            function drawStars() {{
                for (let star of stars) {{
                    let tw = sin(frameCount * star.twinkleSpeed + star.phase) * 0.5 + 0.5;
                    let alpha = map(tw, 0, 1, 30, 200);
                    fill(255, 255, 255, alpha);
                    noStroke();
                    ellipse(star.x + width/2, star.y + height/2, star.size);
                }}
            }}

            function draw() {{
                background(30);
                push();
                translate(translateX, translateY); scale(scaleFactor);

                drawStars();

                stroke(200,200,200,150);
                strokeWeight(1);
                for (let i = 1; i < positions.length; i++) {{
                    let p = positions[i], nearest = -1, minD = Infinity;
                    for (let j = 1; j < positions.length; j++) {{
                        if (j === i) continue;
                        let q = positions[j], d = dist(p.x, p.y, q.x, q.y);
                        if (d < minD) {{ minD = d; nearest = j; }}
                    }}
                    if (nearest >= 0) {{
                        let q = positions[nearest];
                        line(p.x, p.y, q.x, q.y);
                    }}
                }}

                stroke(150,150,150,100);
                strokeWeight(1);
                for (let edge of randomEdges) {{
                    const [i, j] = edge;
                    let p = positions[i], q = positions[j];
                    line(p.x, p.y, q.x, q.y);
                }}
                noStroke();

                textAlign(CENTER, CENTER);
                textLeading(18);
                for (let i = 0; i < positions.length; i++) {{
                    const {{ x, y }} = positions[i];
                    const fontSize = i === 0 ? 18 : 14;
                    const flagSize = fontSize + 8;
                    textSize(flagSize);
                    fill(255);
                    text(flags[i], x, y - fontSize*3);
                    textSize(fontSize);
                    fill(i === 0 ? color(255,200,100) : 255);
                    const lines = wrapText(wishes[i], 20);
                    const lineHeight = fontSize + 4;
                    let blockHeight = lines.length * lineHeight;
                    const padding = 4;
                    for (let li = 0; li < lines.length; li++) {{
                        text(lines[li], x, y - blockHeight/2 + padding + li * lineHeight);
                    }}
                }}
                pop();

                fill(255,255,255,150);
                noStroke();
                ellipse(mouseX, mouseY, 20, 20);
            }}

            function mouseWheel(event) {{
                const zoomFactor = 0.1;
                const zoom = event.delta > 0 ? 1 - zoomFactor : 1 + zoomFactor;
                const wx = (mouseX - translateX) / scaleFactor;
                const wy = (mouseY - translateY) / scaleFactor;
                scaleFactor *= zoom;
                translateX = mouseX - wx * scaleFactor;
                translateY = mouseY - wy * scaleFactor;
                scaleFactor = constrain(scaleFactor, 0.5, 3);
            }}

            function mousePressed() {{
                isDragging = true;
                lastMouseX = mouseX;
                lastMouseY = mouseY;
            }}

            function mouseReleased() {{
                isDragging = false;
            }}

            function mouseDragged() {{
                if (isDragging) {{
                    translateX += mouseX - lastMouseX;
                    translateY += mouseY - lastMouseY;
                    lastMouseX = mouseX;
                    lastMouseY = mouseY;
                }}
            }}

            function windowResized() {{ 
                resizeCanvas(windowWidth, windowHeight); 
            }}
        </script>
      </head>
      <body>
        <main id="canvas-holder"></main>
      </body>
    </html>
    """
    components.html(html_code, height=650)


def main():
    st.title("七夕量子曼荼羅")

    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    # 国情報は従来通り、ローカルのCSVから読み込み
    country_file = os.path.join("src", "data", "country_flags.csv")
    try:
        country_df = pd.read_csv(country_file)
        country_list = country_df["Country (English)"].tolist()
    except FileNotFoundError:
        st.error(f"国情報のCSVファイル '{country_file}' が見つかりません。")
        return

    st.subheader("願い事を入力してください")
    age = st.selectbox("年代", ["-10", "11-20", "21-30", "31-40", "41-50", "51-60", "61-70", "71-"], index=4)
    gender = st.selectbox("性別", ["男性", "女性", "回答しない"], index=0)
    country = st.selectbox("国", country_list, index=country_list.index("Japan"))
    
    prefecture = None
    if country == "Japan":
        prefecture_options = [
            "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
            "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
            "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県",
            "岐阜県", "静岡県", "愛知県", "三重県",
            "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県",
            "鳥取県", "島根県", "岡山県", "広島県", "山口県",
            "徳島県", "香川県", "愛媛県", "高知県",
            "福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県"
        ]
        prefecture = st.selectbox("都道府県", prefecture_options, index=prefecture_options.index("大阪府"))
    
    wish = st.text_input("願い事")
    
    # 願い事を送信するブロック（Google Sheets に書き込み）
    if st.button("願い事を送信"):
        if wish:
            with st.spinner("翻訳中..."):
                target_languages = ['ja', 'en', 'zh-CN', 'ko']
                translations = translate_wish(wish, target_languages)
            
            try:
                flag = country_df.loc[country_df["Country (English)"] == country, "Emoji"].values[0]
            except IndexError:
                flag = ""
            
            new_data = {
                "age": age,
                "gender": gender,
                "country": country,
                "prefecture": prefecture if country == "Japan" else None,
                "wish_original": wish,
                "wish_ja": translations.get('ja', ''),
                "wish_en": translations.get('en', ''),
                "wish_zh": translations.get('zh-CN', ''),
                "wish_ko": translations.get('ko', ''),
                "flag": flag,
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            write_wish(new_data)
            st.success("あなたの願いは、世界中の人の願いと共に銀河に送られました！")
            
            st.session_state.submitted = True
            st.session_state.new_data = new_data
        else:
            st.error("願い事を入力してください。")

    # 願い事送信後、他の願い事を読み込むブロック（Google Sheets から読み込み）
    if st.session_state.get("submitted"):
        st.subheader("銀河に送られた願い事を見る")
        with st.form(key="other_wishes_form"):
            num_wishes = st.slider("読み込む願い事の数", min_value=10, max_value=100, value=20, step=1)
            submit_other_wishes = st.form_submit_button("願い事を読み込む")
        
        if submit_other_wishes:
            with st.spinner("あなたの願い事を銀河の中から探索中..."):
                new_data = st.session_state.new_data
                records = read_wishes(1000)
                df = pd.DataFrame(records)
                filtered_df = df[df["wish_original"] != new_data["wish_original"]]
                other_wishes = filtered_df.sample(n=min(num_wishes, len(filtered_df)), random_state=42)
                with st.spinner("量子の世界と交信中..."):
                    df_result = process_wishes_with_fixed_start(new_data, other_wishes)
            st.session_state.df_result = df_result
            if not df_result.empty:
                st.success("願い事銀河が見つかりました！")
            else:
                st.warning("結果が空です。エラーが発生した可能性があります。")

    # 結果を表示するブロック
    if "df_result" in st.session_state:
        if st.button("願い事銀河を表示"):
            df_result = st.session_state.df_result
            if not df_result.empty:
                wish_original = df_result["wish_original"].tolist()
                ranks = df_result["配置順"].astype(int).tolist()
                flags = df_result.get("flag", ["" for _ in wish_original]).tolist()
                wish_en = df_result.get("wish_en", ["" for _ in wish_original]).tolist()
                wish_ja = df_result.get("wish_ja", ["" for _ in wish_original]).tolist()
                wish_zh = df_result.get("wish_zh", ["" for _ in wish_original]).tolist()
                wish_ko = df_result.get("wish_ko", ["" for _ in wish_original]).tolist()
                render_p5_spiral(wish_original, ranks, flags, wish_en, wish_ja, wish_zh, wish_ko)
            else:
                st.warning("結果データが空です。エラーが発生した可能性があります。")


if __name__ == "__main__":
    main()