import streamlit as st
import pandas as pd
import io
import os
import google.generativeai as genai
from scipy import stats
import altair as alt

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
        padding: 8px 20px;
        border-radius: 5px;
        font-size: 16px;
        font-weight: bold;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    
    .info-box {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
        min-height: 80px;
    }
    .info-label { font-weight: bold; color: #555555 !important; font-size: 14px; margin-bottom: 5px; }
    .info-value { font-weight: bold; color: #000000 !important; font-size: 16px; word-break: break-all; }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    <h1 class="main-title">🚀 기업별 성과 및 지원사업 의사결정 시스템</h1>
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
            col_name = f"{a_s}_{b_s}".strip("_") if b_s and b_s.lower() != 'nan' else a_s
            new_cols.append(col_name.replace(" ", "")) 
        
        f.seek(0)
        df = pd.read_csv(f, skiprows=2, header=None)
        df.columns = new_cols
        
        df['기업명_원본'] = df['기업명'].fillna('')
        df['기업명'] = df['기업명'].fillna('').astype(str).str.strip()
        
        empty_mask = df['기업명'] == ''
        if empty_mask.any():
            df.loc[empty_mask, '기업명'] = [f"(미기재_Row_{i+1})" for i in df[empty_mask].index]
            df.loc[empty_mask, 'is_missing_name'] = True
        else:
            df['is_missing_name'] = False

        text_cols = ['작업담당부서', '원본파일', 'NO.', '전체NO.', '기업명', '기업명_원본', '특이사항', '사업자등록번호', '대표자명', '설립일자', '업종', '사업장 주소', '자기자본전액잠식여부', '바사등급,최종등급수준', 'is_missing_name']
        for col in df.columns:
            if col not in text_cols:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce').fillna(0)
        
        sup_cols = [c for c in df.columns if any(str(i)+'.' in c for i in range(1, 14))]
        df['지원사업_합계'] = df[sup_cols].sum(axis=1).round().astype(int)
        
        for cat in ['매출액', '영업이익', '당기순이익']:
            found_cols = [c for c in df.columns if cat in c]
            cols_3yr = [c for c in found_cols if any(y in c for y in ['2022', '2023', '2024'])]
            if cols_3yr:
                df[f'{cat}_3개년누적'] = df[cols_3yr].sum(axis=1)
            else:
                df[f'{cat}_3개년누적'] = 0
                
        return df
    except Exception as e:
        st.error(f"데이터 처리 오류: {e}")
        return None

def get_col_name(k, y, columns):
    for col in columns:
        if k in col and y in col:
            return col
    return None


# ==============================================================================
# [3] UI 컴포넌트
# ==============================================================================
def render_interactive_table(data, display_cols, col_renames, key_id):
    if data.empty:
        st.info("데이터가 없습니다.")
        return

    df_disp = data.copy()
    df_disp.insert(0, 'No.', range(1, len(df_disp) + 1))
    
    cols_to_use = ['No.', '기업명']
    for c in display_cols:
        if c in df_disp.columns:
            cols_to_use.append(c)
    
    if '지원사업_합계' in df_disp.columns and '지원사업_합계' not in cols_to_use:
        cols_to_use.append('지원사업_합계')

    df_show = df_disp[cols_to_use].copy()
    rename_map = {'No.': 'No.', '기업명': '기업명', '지원사업_합계': '총지원'}
    rename_map.update(col_renames)
    df_show.rename(columns=rename_map, inplace=True)

    for c in df_show.columns:
        if c not in ['No.', '기업명', '총지원'] and '%' not in c:
             if df_show[c].dtype in ['float64', 'int64']:
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
        st.info(f"📁 데이터: {len(df_full)}행")
        exclude_missing = st.checkbox(f"기업명 미기재({missing_count}행) 제외", value=False)
        
        if exclude_missing:
            st.session_state.df = df_full[~df_full['is_missing_name']].copy()
        else:
            st.session_state.df = df_full.copy()

        if st.button("🗑️ 데이터 초기화"):
            st.session_state.df = None
            st.session_state.df_original = None
            st.rerun()
        
        st.divider()
        st.session_state.current_menu = st.radio(
            "📍 메뉴 이동", ["종합 대시보드", "기업 상세 조회", "AI 사업 분석", "📉 심층 통계 분석"],
            index=["종합 대시보드", "기업 상세 조회", "AI 사업 분석", "📉 심층 통계 분석"].index(st.session_state.current_menu)
        )

df = st.session_state.df

if df is not None:
    # --------------------------------------------------------------------------
    # [화면 1] 종합 대시보드
    # --------------------------------------------------------------------------
    if st.session_state.current_menu == "종합 대시보드":
        try:
            st.markdown('<div class="section-header" style="margin-top:0;">🔍 필터 (범위 조회) - (단위: 천원)</div>', unsafe_allow_html=True)
            with st.expander("▼ 금액 범위 설정 (Click)", expanded=False):
                max_rev = int(df['매출액_3개년누적'].max()) if '매출액_3개년누적' in df.columns else 100000000000
                f1, f2, f3 = st.columns(3)
                with f1:
                    f_min_rev = st.number_input("매출 최소", value=0, step=10000, key="min_r")
                    f_max_rev = st.number_input("매출 최대", value=max_rev, step=10000, key="max_r")
                with f2:
                    f_min_op = st.number_input("영업이익 최소", value=-100000000000, step=10000, key="min_o")
                    f_max_op = st.number_input("영업이익 최대", value=100000000000, step=10000, key="max_o")
                with f3:
                    f_min_ni = st.number_input("순이익 최소", value=-100000000000, step=10000, key="min_n")
                    f_max_ni = st.number_input("순이익 최대", value=100000000000, step=10000, key="max_n")

            df_filtered = df.copy()
            if '매출액_3개년누적' in df_filtered.columns:
                df_filtered = df_filtered[(df_filtered['매출액_3개년누적'] >= f_min_rev) & (df_filtered['매출액_3개년누적'] <= f_max_rev)]
            if '영업이익_3개년누적' in df_filtered.columns:
                df_filtered = df_filtered[(df_filtered['영업이익_3개년누적'] >= f_min_op) & (df_filtered['영업이익_3개년누적'] <= f_max_op)]
            if '당기순이익_3개년누적' in df_filtered.columns:
                df_filtered = df_filtered[(df_filtered['당기순이익_3개년누적'] >= f_min_ni) & (df_filtered['당기순이익_3개년누적'] <= f_max_ni)]

            st.success(f"검색된 기업: **{len(df_filtered)}개**")

            r_cols = [get_col_name('매출액', y, df.columns) for y in ['2022', '2023', '2024']]
            o_cols = [get_col_name('영업이익', y, df.columns) for y in ['2022', '2023', '2024']]
            n_cols = [get_col_name('당기순이익', y, df.columns) for y in ['2022', '2023', '2024']]

            st.write("### 📊 성과 요약")
            m1, m2, m3, m4, m5 = st.columns(5)
            
            df_r = df_filtered[df_filtered[r_cols[2]] > df_filtered[r_cols[1]]] if all(r_cols) else pd.DataFrame()
            df_o = df_filtered[df_filtered[o_cols[2]] > df_filtered[o_cols[1]]] if all(o_cols) else pd.DataFrame()
            df_n = df_filtered[df_filtered[n_cols[2]] > df_filtered[n_cols[1]]] if all(n_cols) else pd.DataFrame()

            m1.metric("기업 수", f"{len(df_filtered)}개")
            m2.metric("매출 성장", f"{len(df_r)}개")
            m3.metric("영업이익 성장", f"{len(df_o)}개")
            m4.metric("순이익 증가", f"{len(df_n)}개")
            m5.metric("지원사업 참여", f"{len(df_filtered[df_filtered['지원사업_합계'] > 0])}개")

            st.markdown('<div class="section-header">🏆 3개년 누적 실적 TOP 100 (단위: 천원)</div>', unsafe_allow_html=True)
            r_header = {r_cols[0]:'22년', r_cols[1]:'23년', r_cols[2]:'24년', '매출액_3개년누적':'3년합계'} if all(r_cols) else {}
            o_header = {o_cols[0]:'22년', o_cols[1]:'23년', o_cols[2]:'24년', '영업이익_3개년누적':'3년합계'} if all(o_cols) else {}
            n_header = {n_cols[0]:'22년', n_cols[1]:'23년', n_cols[2]:'24년', '당기순이익_3개년누적':'3년합계'} if all(n_cols) else {}

            c1, c2, c3 = st.columns(3)
            with c1:
                st.write("**매출액 상위**")
                if '매출액_3개년누적' in df_filtered.columns and all(r_cols):
                    render_interactive_table(df_filtered.nlargest(100, '매출액_3개년누적'), r_cols + ['매출액_3개년누적'], r_header, "top_rev")
            with c2:
                st.write("**영업이익 상위**")
                if '영업이익_3개년누적' in df_filtered.columns and all(o_cols):
                    render_interactive_table(df_filtered.nlargest(100, '영업이익_3개년누적'), o_cols + ['영업이익_3개년누적'], o_header, "top_op")
            with c3:
                st.write("**당기순이익 상위**")
                if '당기순이익_3개년누적' in df_filtered.columns and all(n_cols):
                    render_interactive_table(df_filtered.nlargest(100, '당기순이익_3개년누적'), n_cols + ['당기순이익_3개년누적'], n_header, "top_ni")

            st.markdown('<div class="section-header">📈 실적 개선 기업 상세</div>', unsafe_allow_html=True)
            d1, d2, d3 = st.columns(3)
            with d1:
                st.write("**매출 성장 기업**")
                if not df_r.empty: render_interactive_table(df_r, r_cols + ['매출액_3개년누적'], r_header, "imp_rev")
            with d2:
                st.write("**영업이익 성장 기업**")
                if not df_o.empty: render_interactive_table(df_o, o_cols + ['영업이익_3개년누적'], o_header, "imp_op")
            with d3:
                st.write("**당기순이익 증가 기업**")
                if not df_n.empty: render_interactive_table(df_n, n_cols + ['당기순이익_3개년누적'], n_header, "imp_ni")

        except Exception as e:
            st.error(f"대시보드 렌더링 오류: {e}")

    # --------------------------------------------------------------------------
    # [화면 2] 기업 상세 조회
    # --------------------------------------------------------------------------
    elif st.session_state.current_menu == "기업 상세 조회":
        st.subheader("🔍 기업 상세 데이터 조회")
        search_name = st.text_input("기업명 검색", value=st.session_state.target_company)
        
        if search_name:
            search_clean = search_name.strip()
            exact = df[df['기업명'] == search_clean]
            comp_data = exact if not exact.empty else df[df['기업명'].str.contains(search_clean, na=False, regex=False)]
            
            if not comp_data.empty:
                row = comp_data.iloc[0]
                st.markdown('<div class="section-header">🏢 기본 정보</div>', unsafe_allow_html=True)
                info_c1, info_c2, info_c3, info_c4 = st.columns(4)
                
                def get_val(col): return row.get(col, '')
                
                with info_c1:
                    st.markdown(f'<div class="info-box"><div class="info-label">기업명</div><div class="info-value">{row["기업명"]}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="info-box"><div class="info-label">대표자명</div><div class="info-value">{get_val("대표자명")}</div></div>', unsafe_allow_html=True)
                with info_c2:
                    st.markdown(f'<div class="info-box"><div class="info-label">사업자등록번호</div><div class="info-value">{get_val("사업자등록번호")}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="info-box"><div class="info-label">설립일자</div><div class="info-value">{get_val("설립일자")}</div></div>', unsafe_allow_html=True)
                with info_c3:
                    st.markdown(f'<div class="info-box"><div class="info-label">업종</div><div class="info-value">{get_val("업종")}</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="info-box"><div class="info-label">상시종업원수</div><div class="info-value">{int(row.get("상시종업원수",0))}명</div></div>', unsafe_allow_html=True)
                with info_c4:
                    st.markdown(f'<div class="info-box"><div class="info-label">특허건수</div><div class="info-value">{int(row.get("특허건수",0))}건</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="info-box"><div class="info-label">지원사업 합계</div><div class="info-value">{int(row["지원사업_합계"])}건</div></div>', unsafe_allow_html=True)

                st.markdown('<div class="section-header">📢 특이사항</div>', unsafe_allow_html=True)
                remark = row.get('특이사항', '')
                if pd.isna(remark) or str(remark).strip() == 'nan': remark = ""
                st.markdown(f'<div class="info-box"><div class="info-value" style="font-weight:normal; min-height: 20px;">{remark}</div></div>', unsafe_allow_html=True)

                st.markdown('<div class="section-header">📈 재무 현황 (단위: 천원)</div>', unsafe_allow_html=True)
                years = ['2022', '2023', '2024']
                fin_data = {'구분': ['매출액', '영업이익', '당기순이익']}
                for year in years:
                    fin_data[f'{year}년'] = [
                        row.get(get_col_name('매출액', year, df.columns), 0),
                        row.get(get_col_name('영업이익', year, df.columns), 0),
                        row.get(get_col_name('당기순이익', year, df.columns), 0)
                    ]
                st.dataframe(pd.DataFrame(fin_data), use_container_width=True)

                st.markdown('<div class="section-header">🤝 지원사업 참여 상세 내역</div>', unsafe_allow_html=True)
                sup_cols = [c for c in df.columns if any(str(i)+'.' in c for i in range(1, 14))]
                records = []
                for col in sup_cols:
                    if row.get(col, 0) > 0:
                        parts = col.split('_')
                        name = parts[0]
                        yr = parts[1].replace('.0','') if len(parts) > 1 else '기타'
                        records.append({'사업명': name, '연도': yr, '건수': int(row[col])})
                
                if records:
                    df_rec = pd.DataFrame(records)
                    df_pivot = df_rec.pivot_table(index='사업명', columns='연도', values='건수', aggfunc='sum', fill_value=0)
                    df_pivot['합계'] = df_pivot.sum(axis=1)
                    st.dataframe(df_pivot, use_container_width=True)
                else:
                    st.info("참여 내역이 없습니다.")

        st.write("---")
        if st.button("⬅️ 종합 대시보드로 돌아가기"):
            st.session_state.target_company = ""
            st.session_state.current_menu = "종합 대시보드"
            st.rerun()

    # --------------------------------------------------------------------------
    # [화면 3] AI 사업 분석
    # --------------------------------------------------------------------------
    elif st.session_state.current_menu == "AI 사업 분석":
        st.subheader("🤖 Gemini AI: 지원사업별 성과 심층 분석")
        
        all_sup_cols = [c for c in df.columns if any(str(i)+'.' in c for i in range(1, 14))]
        program_names = set([c.split('_')[0] for c in all_sup_cols])
        try: sorted_programs = sorted(list(program_names), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 999)
        except: sorted_programs = sorted(list(program_names))

        selected_program = st.selectbox("분석 대상 지원사업", ["전체 분석"] + sorted_programs)
        api_key = st.text_input("Google Gemini API Key", type="password")
        
        years = ['2022', '2023', '2024']
        r_cols = [get_col_name('매출액', y, df.columns) for y in years]
        o_cols = [get_col_name('영업이익', y, df.columns) for y in years]
        n_cols = [get_col_name('당기순이익', y, df.columns) for y in years]

        if all(r_cols) and all(o_cols) and all(n_cols):
            df_calc = df.copy()
            if selected_program == "전체 분석": target_df = df_calc[df_calc['지원사업_합계'] > 0]
            else: 
                rels = [c for c in all_sup_cols if c.startswith(selected_program)]
                target_df = df_calc[df_calc[rels].sum(axis=1) > 0]

            def get_top_growth(data, col_list, label):
                c22, c23, c24 = col_list
                mask = data[c22] > 0
                temp_df = data[mask].copy()
                temp_df[f'{label}성장률'] = ((temp_df[c24] - temp_df[c22]) / temp_df[c22] * 100).round(1)
                return temp_df.nlargest(10, f'{label}성장률')

            st.markdown(f"### 📊 [{selected_program}] 성장률 TOP 10 (단위: 천원)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**1. 매출액 성장**")
                top_rev = get_top_growth(target_df, r_cols, '매출')
                if not top_rev.empty: 
                    render_interactive_table(top_rev, ['매출성장률', r_cols[2], r_cols[0]], {'매출성장률':'성장률(%)', r_cols[2]:'24년', r_cols[0]:'22년'}, "ai_rev")
                else: st.info("데이터 없음")
            
            with col2:
                st.markdown("**2. 영업이익 성장**")
                top_op = get_top_growth(target_df, o_cols, '영업이익')
                if not top_op.empty:
                    render_interactive_table(top_op, ['영업이익성장률', o_cols[2], o_cols[0]], {'영업이익성장률':'성장률(%)', o_cols[2]:'24년', o_cols[0]:'22년'}, "ai_op")
                else: st.info("데이터 없음")

            with col3:
                st.markdown("**3. 당기순이익 성장**")
                top_ni = get_top_growth(target_df, n_cols, '순이익')
                if not top_ni.empty:
                    render_interactive_table(top_ni, ['순이익성장률', n_cols[2], n_cols[0]], {'순이익성장률':'성장률(%)', n_cols[2]:'24년', n_cols[0]:'22년'}, "ai_ni")
                else: st.info("데이터 없음")

            st.markdown("---")
            if st.button("📊 AI 심층 분석 시작"):
                if not api_key: st.error("API Key를 입력해주세요.")
                elif top_rev.empty and top_op.empty and top_ni.empty: st.error("분석할 데이터가 없습니다.")
                else:
                    try:
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        with st.spinner("분석 중..."):
                            data_context = f"지원사업: {selected_program}\n"
                            if not top_rev.empty: data_context += f"\n[매출 성장 Top 10]\n{top_rev.to_csv(index=False)}"
                            if not top_op.empty: data_context += f"\n[영업이익 성장 Top 10]\n{top_op.to_csv(index=False)}"
                            if not top_ni.empty: data_context += f"\n[순이익 성장 Top 10]\n{top_ni.to_csv(index=False)}"
                            
                            prompt = f"당신은 전문가입니다. {selected_program} 사업 참여 기업의 성과를 분석해주세요.\n{data_context}"
                            response = model.generate_content(prompt)
                            st.markdown(response.text)
                    except Exception as e: st.error(f"오류: {e}")
        else:
            st.error("필수 재무 데이터 컬럼(22, 23, 24년)을 찾을 수 없습니다.")

    # --------------------------------------------------------------------------
    # [화면 4] 심층 통계 분석
    # --------------------------------------------------------------------------
    elif st.session_state.current_menu == "📉 심층 통계 분석":
        st.subheader("📉 심층 통계 분석 (단위: 천원)")
        
        all_sup_cols = [c for c in df.columns if any(str(i)+'.' in c for i in range(1, 14))]
        program_names = set([c.split('_')[0] for c in all_sup_cols])
        try: sorted_programs = sorted(list(program_names), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 999)
        except: sorted_programs = sorted(list(program_names))
        
        target_prog = st.selectbox("분석 대상 사업 선택", ["전체 지원사업 통합"] + sorted_programs)

        r22, r23, r24 = [get_col_name('매출액', y, df.columns) for y in ['2022','2023','2024']]
        o22, o23, o24 = [get_col_name('영업이익', y, df.columns) for y in ['2022','2023','2024']]
        n22, n23, n24 = [get_col_name('당기순이익', y, df.columns) for y in ['2022','2023','2024']]

        if all([r22,r24,o22,o24,n22,n24]):
            df_stat = df.copy()
            # 성장률
            df_stat['매출성장률'] = df_stat.apply(lambda x: ((x[r24]-x[r22])/x[r22]*100) if x[r22]>0 else 0, axis=1)
            df_stat['영업이익성장률'] = df_stat.apply(lambda x: ((x[o24]-x[o22])/abs(x[o22])*100) if abs(x[o22])>0 else 0, axis=1)
            df_stat['순이익성장률'] = df_stat.apply(lambda x: ((x[n24]-x[n22])/abs(x[n22])*100) if abs(x[n22])>0 else 0, axis=1)

            # 타겟 그룹 정의
            if target_prog == "전체 지원사업 통합":
                df_stat['is_target'] = df_stat['지원사업_합계'] > 0
                rels = []
            else:
                rels = [c for c in all_sup_cols if c.startswith(target_prog)]
                df_stat['is_target'] = df_stat[rels].sum(axis=1) > 0
                df_stat['해당사업지원수'] = df_stat[rels].sum(axis=1)

            group_yes = df_stat[df_stat['is_target']]
            group_no = df_stat[~df_stat['is_target']]

            # 1. 통계적 유의 검증
            st.markdown('<div class="section-header">1️⃣ 통계적 유의 검증 (참여 vs 미참여)</div>', unsafe_allow_html=True)
            
            def check_significance(col, name):
                if len(group_yes) > 1 and len(group_no) > 1:
                    t, p = stats.ttest_ind(group_yes[col], group_no[col], equal_var=False)
                    res = "✅ 유의미" if p < 0.05 else "⚠️ 유의미하지 않음"
                    return f"{group_yes[col].mean():.1f}", f"{group_no[col].mean():.1f}", res, p
                return "-", "-", "-", 1.0

            if target_prog == "전체 지원사업 통합":
                sup_mean = group_yes['지원사업_합계'].mean()
                sup_res = f"평균 {sup_mean:.1f}회"
            else:
                s1, s2, s3, s4 = check_significance('지원사업_합계', '총 지원회수')
                sup_res = f"{s3} (p={s4:.3f})"

            r_m1, r_m2, r_res, r_p = check_significance('매출성장률', '매출')
            o_m1, o_m2, o_res, o_p = check_significance('영업이익성장률', '영업이익')
            n_m1, n_m2, n_res, n_p = check_significance('순이익성장률', '순이익')

            stat_data = {
                '구분': ['매출액 성장률', '영업이익 성장률', '당기순이익 성장률', '총 지원회수'],
                '참여기업 평균': [f"{r_m1}%", f"{o_m1}%", f"{n_m1}%", f"{group_yes['지원사업_합계'].mean():.1f}회"],
                '미참여기업 평균': [f"{r_m2}%", f"{o_m2}%", f"{n_m2}%", f"{group_no['지원사업_합계'].mean():.1f}회"],
                '검증 결과': [f"{r_res} (p={r_p:.3f})", f"{o_res} (p={o_p:.3f})", f"{n_res} (p={n_p:.3f})", sup_res]
            }
            st.dataframe(pd.DataFrame(stat_data), use_container_width=True)

            # 2. 사업 포트폴리오 매트릭스
            st.markdown('<div class="section-header">2️⃣ 사업 포트폴리오 매트릭스</div>', unsafe_allow_html=True)
            if target_prog == "전체 지원사업 통합":
                matrix_data = []
                for prog in sorted_programs:
                    rls = [c for c in all_sup_cols if c.startswith(prog)]
                    s_df = df_stat[df_stat[rls].sum(axis=1) > 0]
                    if len(s_df) > 0:
                        matrix_data.append({'사업명': prog, '참여수': len(s_df), '성장률': s_df['매출성장률'].mean()})
                
                if matrix_data:
                    chart = alt.Chart(pd.DataFrame(matrix_data)).mark_circle(size=100).encode(
                        x='참여수', y='성장률', color='사업명', tooltip=['사업명', '참여수', '성장률']
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
                    
                    # 3. 하단 상세 리스트
                    st.markdown("#### ▼ 포트폴리오 상세 조회")
                    sel_mat_prog = st.selectbox("확인할 사업을 선택하세요", sorted_programs, key="mat_sel")
                    
                    rls_sub = [c for c in all_sup_cols if c.startswith(sel_mat_prog)]
                    df_stat['선택사업_지원수'] = df_stat[rls_sub].sum(axis=1)
                    target_list = df_stat[df_stat['선택사업_지원수'] > 0].copy()
                    
                    st.write(f"**'{sel_mat_prog}' 참여 기업 리스트 ({len(target_list)}개)**")
                    if not target_list.empty:
                        cols_to_display = [r22, r23, r24, o22, o23, o24, n22, n23, n24, '선택사업_지원수']
                        rename_rule = {
                            r22:'매출(22)', r23:'매출(23)', r24:'매출(24)',
                            o22:'영업(22)', o23:'영업(23)', o24:'영업(24)',
                            n22:'순익(22)', n23:'순익(23)', n24:'순익(24)',
                            '선택사업_지원수': '지원사업참여회수'
                        }
                        render_interactive_table(target_list, cols_to_display, rename_rule, "mat_table")
            else:
                st.info("전체 통합 모드에서만 포트폴리오가 활성화됩니다.")
            
            st.write("---")
            
            # 3. 상관관계 분석
            st.markdown('<div class="section-header">3️⃣ 지원 횟수와 성장의 상관관계 분석</div>', unsafe_allow_html=True)
            if len(group_yes) > 5: 
                y_option = st.selectbox("분석할 성과 지표", ['매출성장률', '영업이익성장률', '순이익성장률'])
                x_col = '해당사업지원수' if target_prog != "전체 지원사업 통합" else '지원사업_합계'
                
                chart_df = group_yes[['기업명', x_col, y_option]].copy()
                corr_coef, p_val_corr = stats.pearsonr(chart_df[x_col], chart_df[y_option])
                slope, intercept, r_value, p_value, std_err = stats.linregress(chart_df[x_col], chart_df[y_option])

                c1, c2 = st.columns(2)
                with c1: st.metric("상관계수 (R)", f"{corr_coef:.3f}")
                with c2: st.metric("회귀계수 (기울기)", f"{slope:.2f}")

                scatter = alt.Chart(chart_df).mark_circle(size=60).encode(
                    x=alt.X(x_col, title='지원 횟수'), y=alt.Y(y_option, title=f'{y_option} (%)'), tooltip=['기업명', x_col, y_option]
                )
                reg_line = scatter.transform_regression(x_col, y_option).mark_line(color='red')
                st.altair_chart((scatter + reg_line).interactive(), use_container_width=True)
            else:
                st.error("분석 표본이 부족합니다.")

    st.write("---")
    _, col_exit = st.columns([15, 1])
    with col_exit:
        if st.button("종료"):
            os._exit(0)

else:
    st.info("왼쪽 사이드바에서 데이터 파일을 업로드해 주세요.")