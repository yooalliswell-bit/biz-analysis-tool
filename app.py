import streamlit as st
import pandas as pd
import io
import os
import google.generativeai as genai

# ==============================================================================
# [1] 페이지 설정 및 스타일
# ==============================================================================
st.set_page_config(page_title="기업 성과 분석 시스템", layout="wide")

if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'current_menu' not in st.session_state:
    st.session_state.current_menu = "종합 대시보드"
if 'target_company' not in st.session_state:
    st.session_state.target_company = ""

st.markdown("""
    <style>
    .main-title { text-align: center; color: #1E1E1E; margin-bottom: 30px; font-weight: bold; }
    
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        padding: 20px !important;
        border-radius: 12px !important;
        border: 2px solid #eeeeee !important;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stMetricLabel"] p { color: #000000 !important; font-size: 16px !important; font-weight: bold !important; }
    [data-testid="stMetricValue"] div { color: #000000 !important; font-weight: 800 !important; }
    
    .section-header {
        background-color: #1E1E1E;
        color: #FFFFFF;
        padding: 12px 20px;
        border-radius: 5px;
        font-size: 18px;
        font-weight: bold;
        margin-top: 35px;
        margin-bottom: 15px;
    }
    
    .info-box {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }
    .info-label { font-weight: bold; color: #555555 !important; font-size: 14px; }
    .info-value { font-weight: bold; color: #000000 !important; font-size: 18px; margin-bottom: 5px; }
    
    /* 경고 박스 스타일 */
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    <h1 class="main-title">🚀 기업별 성과 및 지원사업 분석 시스템</h1>
    """, unsafe_allow_html=True)


# ==============================================================================
# [2] 데이터 처리 엔진
# ==============================================================================
def load_and_process_data(file):
    try:
        raw_bytes = file.read()
        try:
            content = raw_bytes.decode('utf-8-sig')
        except UnicodeDecodeError:
            content = raw_bytes.decode('cp949')
        
        f = io.StringIO(content)
        header_df = pd.read_csv(f, header=None, nrows=2)
        h0 = header_df.iloc[0].ffill().tolist()
        h1 = header_df.iloc[1].fillna("").tolist()
        
        new_cols = []
        for a, b in zip(h0, h1):
            a_s, b_s = str(a).strip(), str(b).strip()
            new_cols.append(f"{a_s}_{b_s}".strip("_") if b_s and b_s.lower() != 'nan' else a_s)
        
        f.seek(0)
        df = pd.read_csv(f, skiprows=2, header=None)
        df.columns = new_cols
        
        # [데이터 전처리] 기업명 정리
        df['기업명_원본'] = df['기업명'].fillna('')
        df['기업명'] = df['기업명'].fillna('').astype(str).str.strip()
        
        # 기업명이 없는 행에 임시 ID 부여
        empty_mask = df['기업명'] == ''
        if empty_mask.any():
            df.loc[empty_mask, '기업명'] = [f"(미기재_Row_{i+1})" for i in df[empty_mask].index]
            df.loc[empty_mask, 'is_missing_name'] = True
        else:
            df['is_missing_name'] = False

        # 숫자 강제 변환
        text_cols = ['작업담당부서', '원본파일', 'NO.', '전체NO.', '기업명', '기업명_원본', '특이사항', '사업자등록번호', '대표자명', '설립일자', '업종', '사업장 주소', '자기자본 전액 잠식 여부', '바사등급, 최종등급 수준', 'is_missing_name']
        for col in df.columns:
            if col not in text_cols:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0)
        
        # 지원사업 합계
        sup_cols = [c for c in df.columns if any(str(i)+'.' in c for i in range(1, 14))]
        df['지원사업_합계'] = df[sup_cols].sum(axis=1).round().astype(int)
        
        # 3개년 누적 계산
        for cat in ['매출액', '영업이익', '당기순이익']:
            cols_3yr = [c for c in df.columns if cat in c and any(y in c for y in ['2022', '2023', '2024'])]
            if cols_3yr:
                df[f'{cat}_3개년누적'] = df[cols_3yr].sum(axis=1)
                
        return df
    except Exception as e:
        st.error(f"데이터 처리 오류: {e}")
        return None


# ==============================================================================
# [3] UI 컴포넌트
# ==============================================================================
def render_interactive_table(data, display_cols, col_renames, key_id):
    if data.empty:
        st.info("조건에 맞는 데이터가 없습니다.")
        return

    df_disp = data.copy()
    df_disp.insert(0, 'No.', range(1, len(df_disp) + 1))
    
    target_cols = ['No.', '기업명'] + display_cols + ['지원사업_합계']
    valid_cols = [c for c in target_cols if c in df_disp.columns]
    df_show = df_disp[valid_cols].copy()
    
    rename_map = {'No.': 'No.', '기업명': '기업명', '지원사업_합계': '지원사업'}
    rename_map.update(col_renames)
    df_show.rename(columns=rename_map, inplace=True)

    for c in df_show.columns:
        if c not in ['No.', '기업명', '지원사업']:
             df_show[c] = df_show[c].apply(lambda x: f"{int(x):,}")

    selection = st.dataframe(
        df_show,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=key_id
    )

    if selection.selection.rows:
        selected_idx = selection.selection.rows[0]
        selected_company = df_show.iloc[selected_idx]['기업명']
        st.session_state.target_company = selected_company
        st.session_state.current_menu = "기업 상세 조회"
        st.rerun()


# ==============================================================================
# [4] 메인 로직
# ==============================================================================

with st.sidebar:
    st.header("⚙️ 설정")
    
    if st.session_state.df_original is None:
        uploaded_file = st.file_uploader("CSV 파일 업로드", type=['csv'])
        if uploaded_file:
            with st.spinner("분석 중..."):
                processed_df = load_and_process_data(uploaded_file)
                if processed_df is not None:
                    st.session_state.df_original = processed_df
                    st.session_state.df = processed_df
                    st.success("로드 완료")
                    st.rerun()
    else:
        df_full = st.session_state.df_original
        missing_count = len(df_full[df_full['is_missing_name']])
        
        st.info(f"📁 총 데이터 행: {len(df_full)}개")
        exclude_missing = st.checkbox(f"기업명 없는 행({missing_count}개) 제외하기", value=False)
        
        if exclude_missing:
            st.session_state.df = df_full[~df_full['is_missing_name']].copy()
        else:
            st.session_state.df = df_full.copy()

        if st.button("🗑️ 데이터 초기화"):
            st.session_state.df = None
            st.session_state.df_original = None
            st.session_state.target_company = ""
            st.rerun()
        
        st.divider()
        st.session_state.current_menu = st.radio(
            "📍 메뉴", ["종합 대시보드", "기업 상세 조회", "AI 사업 분석"],
            index=0 if st.session_state.current_menu == "종합 대시보드" else (1 if st.session_state.current_menu == "기업 상세 조회" else 2)
        )

df = st.session_state.df

if df is not None:
    # --------------------------------------------------------------------------
    # [화면 1] 종합 대시보드
    # --------------------------------------------------------------------------
    if st.session_state.current_menu == "종합 대시보드":
        try:
            missing_but_active = df[df['is_missing_name'] & (df['지원사업_합계'] > 0)]
            if not missing_but_active.empty:
                with st.expander("🚨 데이터 차이 원인 분석", expanded=True):
                    st.markdown(f"""
                    <div class="warning-box">
                        <b>[분석 결과]</b> 기업명 누락된 <b>{len(missing_but_active)}개 행</b>에서 지원사업 실적이 발견되었습니다.<br>
                        이 때문에 "지원사업 참여 기업 수"가 <b>{len(missing_but_active)}개</b> 더 많게 나올 수 있습니다.<br>
                        (※ 이 행들이 합계 행이라면 <b>왼쪽 사이드바의 [제외하기]</b>를 체크하세요.)
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(missing_but_active[['기업명', '지원사업_합계', '매출액_3개년누적']], hide_index=True)

            st.markdown('<div class="section-header" style="margin-top:0;">🔍 필터 (범위 조회)</div>', unsafe_allow_html=True)
            with st.expander("▼ 금액 범위 설정 (Click)", expanded=False):
                max_rev = int(df['매출액_3개년누적'].max()) if '매출액_3개년누적' in df.columns else 100000000000
                max_op = int(df['영업이익_3개년누적'].max()) if '영업이익_3개년누적' in df.columns else 100000000000
                max_ni = int(df['당기순이익_3개년누적'].max()) if '당기순이익_3개년누적' in df.columns else 100000000000
                min_op = int(df['영업이익_3개년누적'].min()) if '영업이익_3개년누적' in df.columns else -100000000000
                min_ni = int(df['당기순이익_3개년누적'].min()) if '당기순이익_3개년누적' in df.columns else -100000000000

                f1, f2, f3 = st.columns(3)
                with f1:
                    f_min_rev = st.number_input("매출 최소", value=0, step=10000000, key="min_r")
                    f_max_rev = st.number_input("매출 최대", value=max_rev, step=10000000, key="max_r")
                with f2:
                    f_min_op = st.number_input("영업이익 최소", value=min_op, step=10000000, key="min_o")
                    f_max_op = st.number_input("영업이익 최대", value=max_op, step=10000000, key="max_o")
                with f3:
                    f_min_ni = st.number_input("당기순이익 최소", value=min_ni, step=10000000, key="min_n")
                    f_max_ni = st.number_input("당기순이익 최대", value=max_ni, step=10000000, key="max_n")

            df_filtered = df.copy()
            if '매출액_3개년누적' in df_filtered.columns:
                df_filtered = df_filtered[(df_filtered['매출액_3개년누적'] >= f_min_rev) & (df_filtered['매출액_3개년누적'] <= f_max_rev)]
            if '영업이익_3개년누적' in df_filtered.columns:
                df_filtered = df_filtered[(df_filtered['영업이익_3개년누적'] >= f_min_op) & (df_filtered['영업이익_3개년누적'] <= f_max_op)]
            if '당기순이익_3개년누적' in df_filtered.columns:
                df_filtered = df_filtered[(df_filtered['당기순이익_3개년누적'] >= f_min_ni) & (df_filtered['당기순이익_3개년누적'] <= f_max_ni)]

            st.success(f"조건 만족 기업: **{len(df_filtered)}개**")

            def get_col(k, y):
                matches = [c for c in df.columns if k in c and y in c]
                return matches[0] if matches else None

            r_cols = [get_col('매출액', y) for y in ['2022', '2023', '2024']]
            o_cols = [get_col('영업이익', y) for y in ['2022', '2023', '2024']]
            n_cols = [get_col('당기순이익', y) for y in ['2022', '2023', '2024']]

            st.write("### 📊 성과 요약")
            m1, m2, m3, m4, m5 = st.columns(5)
            
            df_r = df_filtered[df_filtered[r_cols[2]] > df_filtered[r_cols[1]]].copy() if r_cols[2] else pd.DataFrame()
            df_o = df_filtered[df_filtered[o_cols[2]] > df_filtered[o_cols[1]]].copy() if o_cols[2] else pd.DataFrame()
            df_n = df_filtered[df_filtered[n_cols[2]] > df_filtered[n_cols[1]]].copy() if n_cols[2] else pd.DataFrame()

            m1.metric("분석 대상 기업(행) 수", f"{len(df_filtered)}개")
            m2.metric("매출 성장", f"{len(df_r)}개")
            m3.metric("영업이익 성장", f"{len(df_o)}개")
            m4.metric("당기순이익 증가", f"{len(df_n)}개")
            m5.metric("지원사업 참여", f"{len(df_filtered[df_filtered['지원사업_합계'] > 0])}개")

            st.markdown('<div class="section-header">🏆 3개년 누적 실적 TOP 100 (연도별 및 합계 포함)</div>', unsafe_allow_html=True)
            r_header = {r_cols[0]:'22년', r_cols[1]:'23년', r_cols[2]:'24년', '매출액_3개년누적':'3년합계'}
            o_header = {o_cols[0]:'22년', o_cols[1]:'23년', o_cols[2]:'24년', '영업이익_3개년누적':'3년합계'}
            n_header = {n_cols[0]:'22년', n_cols[1]:'23년', n_cols[2]:'24년', '당기순이익_3개년누적':'3년합계'}

            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**매출액 상위 100**")
                if '매출액_3개년누적' in df_filtered.columns:
                    render_interactive_table(df_filtered.nlargest(100, '매출액_3개년누적'), r_cols + ['매출액_3개년누적'], r_header, "top_rev")
            with c2:
                st.write("**영업이익 상위 100**")
                if '영업이익_3개년누적' in df_filtered.columns:
                    render_interactive_table(df_filtered.nlargest(100, '영업이익_3개년누적'), o_cols + ['영업이익_3개년누적'], o_header, "top_op")
            with c3:
                st.write("**당기순이익 상위 100**")
                if '당기순이익_3개년누적' in df_filtered.columns:
                    render_interactive_table(df_filtered.nlargest(100, '당기순이익_3개년누적'), n_cols + ['당기순이익_3개년누적'], n_header, "top_ni")

            st.markdown('<div class="section-header">📈 실적 개선 기업 상세</div>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            with d1:
                st.write("**매출 성장 기업**")
                render_interactive_table(df_r, r_cols + ['매출액_3개년누적'], r_header, "imp_rev")
            with d2:
                st.write("**영업이익 성장 기업**")
                render_interactive_table(df_o, o_cols + ['영업이익_3개년누적'], o_header, "imp_op")
            with d3:
                st.write("**당기순이익 증가 기업**")
                render_interactive_table(df_n, n_cols + ['당기순이익_3개년누적'], n_header, "imp_ni")

        except Exception as e:
            st.error(f"오류: {e}")

    # --------------------------------------------------------------------------
    # [화면 2] 기업 상세 조회
    # --------------------------------------------------------------------------
    elif st.session_state.current_menu == "기업 상세 조회":
        st.subheader("🔍 기업 상세 데이터 조회")
        search_name = st.text_input("기업명 검색", value=st.session_state.target_company)
        
        if search_name:
            search_clean = search_name.strip()
            exact = df[df['기업명'] == search_clean]
            if not exact.empty:
                comp_data = exact
            else:
                comp_data = df[df['기업명'].str.contains(search_clean, na=False, regex=False)]
            
            if not comp_data.empty:
                row = comp_data.iloc[0]
                st.markdown('<div class="section-header">🏢 기본 정보</div>', unsafe_allow_html=True)
                info_c1, info_c2, info_c3, info_c4 = st.columns(4)
                with info_c1:
                    st.markdown(f'<div class="info-box"><div class="info-label">기업명</div><div class="info-value">{row["기업명"]}</div><div class="info-label">대표자명</div><div class="info-value">{row["대표자명"]}</div></div>', unsafe_allow_html=True)
                with info_c2:
                    st.markdown(f'<div class="info-box"><div class="info-label">사업자등록번호</div><div class="info-value">{row["사업자등록번호"]}</div><div class="info-label">설립일자</div><div class="info-value">{row["설립일자"]}</div></div>', unsafe_allow_html=True)
                with info_c3:
                    st.markdown(f'<div class="info-box"><div class="info-label">업종</div><div class="info-value">{row["업종"]}</div><div class="info-label">상시종업원수</div><div class="info-value">{int(row.get("상시종업원수", 0))}명</div></div>', unsafe_allow_html=True)
                with info_c4:
                    st.markdown(f'<div class="info-box"><div class="info-label">특허건수</div><div class="info-value">{int(row.get("특허건수", 0))}건</div><div class="info-label" style="color:blue!important;">지원사업 참여수</div><div class="info-value" style="color:blue!important;">{int(row["지원사업_합계"])}건</div></div>', unsafe_allow_html=True)

                st.markdown('<div class="section-header">📈 연도별 재무 현황 (단위: 원)</div>', unsafe_allow_html=True)
                years = ['2022', '2023', '2024']
                fin_data = {'구분': ['총자산', '자기자본', '매출액', '영업이익', '당기순이익']}
                for year in years:
                    fin_data[f'{year}년'] = [
                        row.get(f'총자산_{year}년', 0), row.get(f'자기자본_{year}년', 0),
                        row.get(f'매출액_{year}년', 0), row.get(f'영업이익_{year}년', 0),
                        row.get(f'당기순이익_{year}년', 0)
                    ]
                fin_df = pd.DataFrame(fin_data)
                for y in years:
                    fin_df[f'{y}년'] = fin_df[f'{y}년'].apply(lambda x: f"{int(x):,}")
                st.dataframe(fin_df, use_container_width=True, hide_index=True)

                st.markdown('<div class="section-header">🤝 지원사업 참여 상세 내역</div>', unsafe_allow_html=True)
                sup_cols = [c for c in df.columns if any(str(i)+'.' in c for i in range(1, 14))]
                records = []
                all_years = set()
                for col in sup_cols:
                    val = row.get(col, 0)
                    try:
                        parts = col.rsplit('_', 1) 
                        if len(parts) == 2:
                            prog_name = parts[0]
                            year = parts[1].replace('.0', '')
                            if val > 0:
                                records.append({'사업명': prog_name, '연도': year, '건수': int(val)})
                                all_years.add(year)
                    except: continue

                if records:
                    df_rec = pd.DataFrame(records)
                    df_pivot = df_rec.pivot_table(index='사업명', columns='연도', values='건수', aggfunc='sum', fill_value=0)
                    df_pivot['합계'] = df_pivot.sum(axis=1)
                    if not df_pivot.empty:
                        sorted_years = sorted(list(all_years))
                        final_cols = sorted_years + ['합계']
                        final_cols = [c for c in final_cols if c in df_pivot.columns]
                        st.dataframe(df_pivot[final_cols], use_container_width=True)
                    else: st.info("내역 없음")
                else: st.info("내역 없음")
            else:
                st.warning(f"'{search_clean}'에 대한 검색 결과가 없습니다.")
        st.write("---")
        if st.button("⬅️ 종합 대시보드로 돌아가기"):
            st.session_state.target_company = ""
            st.session_state.current_menu = "종합 대시보드"
            st.rerun()

    # --------------------------------------------------------------------------
    # [화면 3] AI 사업 분석 (Gemini-1.5-flash 적용)
    # --------------------------------------------------------------------------
    elif st.session_state.current_menu == "AI 사업 분석":
        st.subheader("🤖 Gemini AI: 지원사업 성과 심층 분석")
        
        def get_col_name(k, y):
            m = [c for c in df.columns if k in c and y in c]
            return m[0] if m else None

        r22, r24 = get_col_name('매출액', '2022'), get_col_name('매출액', '2024')
        o22, o24 = get_col_name('영업이익', '2022'), get_col_name('영업이익', '2024')

        api_key = st.text_input("Google Gemini API Key", type="password")

        st.markdown("### 1️⃣ 지원사업 참여 vs 미참여 그룹 비교")
        if r24 and r22:
            df_calc = df.copy()
            mask_valid = df_calc[r22] > 0
            df_calc.loc[mask_valid, '매출성장률'] = (df_calc[r24] - df_calc[r22]) / df_calc[r22] * 100
            
            supported = df_calc[df_calc['지원사업_합계'] > 0]
            not_supported = df_calc[df_calc['지원사업_합계'] == 0]
            
            avg_rev_grow_sup = supported['매출성장률'].mean()
            avg_rev_grow_not = not_supported['매출성장률'].mean()
            
            c1, c2 = st.columns(2)
            c1.metric("지원사업 참여기업 평균 매출성장률", f"{avg_rev_grow_sup:.1f}%")
            c2.metric("미참여기업 평균 매출성장률", f"{avg_rev_grow_not:.1f}%", delta=f"{avg_rev_grow_sup - avg_rev_grow_not:.1f}%p 차이")

        st.markdown("### 2️⃣ AI 심층 분석 요청")
        user_query = st.text_area("추가 요청 사항 (선택)", height=70)
        
        if st.button("📊 지원사업 효과성 집중 분석 시작"):
            if not api_key:
                st.error("API Key를 입력해주세요.")
            else:
                try:
                    genai.configure(api_key=api_key)
                    # [수정됨] 최신 모델 사용
                    model = genai.GenerativeModel('gemini-2.0-flash')
                    
                    with st.spinner("AI 분석 중..."):
                        df_llm = df_calc[['업종', '지원사업_합계', '매출성장률', '매출액_3개년누적']].copy().fillna(0)
                        df_llm['Company_ID'] = [f"IDX_{i}" for i in range(len(df_llm))]
                        
                        csv_data = df_llm.head(100).to_csv(index=False)
                        
                        stats_summary = f"""
                        [통계 요약]
                        - 참여 기업 수: {len(supported)} / 미참여 기업 수: {len(not_supported)}
                        - 참여 기업 평균 매출 성장률: {avg_rev_grow_sup:.1f}%
                        - 미참여 기업 평균 매출 성장률: {avg_rev_grow_not:.1f}%
                        """

                        prompt = f"""
                        당신은 정부 지원사업 성과 분석 전문가입니다.
                        목표: 지원사업 참여가 기업 성장에 기여했는지 분석하세요.

                        [통계 데이터]
                        {stats_summary}

                        [개별 기업 샘플 (상위 100개)]
                        {csv_data}

                        [사용자 요청]
                        {user_query}

                        [작성 가이드]
                        1. 통계 수치를 근거로 지원사업의 효과성을 평가하세요.
                        2. IDX를 사용하여 구체적인 우수 사례를 언급하세요.
                        """
                        
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                        
                        st.markdown("---")
                        id_map = pd.DataFrame({'Company_ID': df_llm['Company_ID'], '실제기업명': df['기업명']})
                        st.dataframe(id_map.head(100), use_container_width=True)

                except Exception as e:
                    st.error(f"분석 중 오류: {e}")

    st.write("---")
    _, col_exit = st.columns([15, 1])
    with col_exit:
        if st.button("종료"):
            os._exit(0)

else:
    st.info("왼쪽 사이드바에서 데이터 파일을 업로드해 주세요.")