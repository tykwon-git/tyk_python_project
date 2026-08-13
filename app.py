
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import rankdata

# ============================================================
# 기온에 따른 서울시 따릉이·지하철 이용 분석 Dashboard patch 05:04
# ============================================================

st.set_page_config(
    page_title="기온에 따른 서울시 교통 이용 분석",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FONT_FAMILY = "Malgun Gothic"
BICYCLE_COLOR = "#2E7D32"
SUBWAY_COLOR = "#1565C0"
NAVY = "#1C283A"
GRAY = "#6B7280"
LIGHT_BG = "#F4F6F9"
BORDER = "#D9DEE5"

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid")

TIME_ORDER = [
    "06-07", "07-08", "08-09", "09-10", "10-11", "11-12",
    "12-13", "13-14", "14-15", "15-16", "16-17", "17-18",
    "18-19", "19-20", "20-21", "21-22", "22-23", "23-24"
]
MONTH_ORDER = [3, 6, 9, 12]
MONTH_LABEL = {3: "3월", 6: "6월", 9: "9월", 12: "12월"}


@st.cache_data
def load_data():
    candidates = [
        Path("data/processed/analysis_dataset.csv"),
        Path("data\\processed\\analysis_dataset.csv"),
    ]

    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "analysis_dataset.csv를 찾을 수 없습니다. "
            "기본 경로: data/processed/analysis_dataset.csv"
        )

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_cols = [
        "날짜", "시간대", "따릉이대여건수", "지하철승차인원",
        "기온(°C)", "요일", "요일명", "주말여부", "공휴일여부"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_cols}")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    for col in ["기온(°C)", "따릉이대여건수", "지하철승차인원"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["월"] = df["날짜"].dt.month
    df["시간대"] = pd.Categorical(
        df["시간대"], categories=TIME_ORDER, ordered=True
    )

    return df.dropna(
        subset=["날짜", "기온(°C)", "따릉이대여건수", "지하철승차인원"]
    ).copy()


df = load_data()


def spearman_corr(x, y):
    """Rank -> Pearson 방식으로 Spearman 상관계수를 계산."""
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)

    valid = x.notna() & y.notna()
    x = x[valid]
    y = y[valid]

    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan

    xr = rankdata(x, method="average")
    yr = rankdata(y, method="average")
    return np.corrcoef(xr, yr)[0, 1]


def calc_usage_index(data, group_col):
    """선택 조건의 전체 평균 이용량=100으로 두고 그룹별 상대 이용지수 계산."""
    if data.empty:
        return pd.DataFrame(columns=[group_col, "따릉이지수", "지하철지수"])

    overall_bicycle = data["따릉이대여건수"].mean()
    overall_subway = data["지하철승차인원"].mean()

    summary = (
        data.groupby(group_col, observed=False)[
            ["따릉이대여건수", "지하철승차인원"]
        ]
        .mean()
        .reset_index()
    )

    summary["따릉이지수"] = (
        summary["따릉이대여건수"] / overall_bicycle * 100
        if overall_bicycle != 0 else np.nan
    )
    summary["지하철지수"] = (
        summary["지하철승차인원"] / overall_subway * 100
        if overall_subway != 0 else np.nan
    )
    return summary


# ============================================================
# CSS
# ============================================================
st.markdown(
    f"""
    <style>
    .main-title {{
        font-size: 2.1rem;
        font-weight: 800;
        color: {NAVY};
        margin-bottom: 0.2rem;
    }}
    .sub-title {{
        color: {GRAY};
        font-size: 0.98rem;
        margin-bottom: 1.1rem;
    }}
    .section-title {{
        font-size: 1.35rem;
        font-weight: 750;
        color: {NAVY};
        margin-top: 0.75rem;
        margin-bottom: 0.2rem;
    }}
    .section-desc {{
        font-size: 0.9rem;
        color: {GRAY};
        margin-bottom: 0.6rem;
    }}
    .kpi-card {{
        background: white;
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 0.8rem 1rem;
        min-height: 120px;
    }}
    .kpi-label {{
        font-size: 0.85rem;
        color: {GRAY};
        margin-bottom: 0.35rem;
    }}
    .kpi-value {{
        font-size: 1.55rem;
        font-weight: 800;
        color: {NAVY};
    }}
    .kpi-help {{
        font-size: 0.72rem;
        color: {GRAY};
        margin-top: 0.25rem;
    }}
    .notice {{
        background: {LIGHT_BG};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        color: {NAVY};
        font-size: 0.82rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Title
# ============================================================
st.markdown(
    '<div class="main-title">기온에 따른 서울시 따릉이·지하철 이용 분석</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">선택한 기간·기온·요일 조건에서 두 이동수단의 이용 패턴과 기온과의 관계를 탐색합니다.</div>',
    unsafe_allow_html=True,
)

# ============================================================
# Filters
# ============================================================
f1, f2, f3, f4, f5 = st.columns([2.2, 2.1, 1.8, 2.0, 2.2])

min_date = df["날짜"].min().date()
max_date = df["날짜"].max().date()

with f1:
    date_range = st.date_input(
        "기간",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

with f2:
    temp_min = float(np.floor(df["기온(°C)"].min()))
    temp_max = float(np.ceil(df["기온(°C)"].max()))
    temp_range = st.slider(
        "기온 범위 (°C)",
        min_value=temp_min,
        max_value=temp_max,
        value=(temp_min, temp_max),
        step=1.0,
    )

with f3:
    day_type = st.selectbox(
        "이용일 유형",
        ["평일·비공휴일", "전체", "주말", "공휴일"],
        index=0,
    )

with f4:
    selected_months = st.multiselect(
        "월",
        options=MONTH_ORDER,
        default=MONTH_ORDER,
        format_func=lambda x: MONTH_LABEL[x],
    )

with f5:
    selected_hours = st.multiselect(
        "시간대",
        options=TIME_ORDER,
        default=TIME_ORDER,
    )

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

filtered = df[
    (df["날짜"].dt.date >= start_date)
    & (df["날짜"].dt.date <= end_date)
    & (df["기온(°C)"] >= temp_range[0])
    & (df["기온(°C)"] <= temp_range[1])
    & (df["월"].isin(selected_months))
    & (df["시간대"].isin(selected_hours))
].copy()

if day_type == "평일·비공휴일":
    filtered = filtered[
        (~filtered["주말여부"]) & (~filtered["공휴일여부"])
    ].copy()
elif day_type == "주말":
    filtered = filtered[filtered["주말여부"]].copy()
elif day_type == "공휴일":
    filtered = filtered[filtered["공휴일여부"]].copy()

st.markdown(
    f'<div class="notice">현재 조건: {len(filtered):,}개 관측값 · '
    f'{start_date} ~ {end_date} · 기온 {temp_range[0]:.0f}~{temp_range[1]:.0f}°C</div>',
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("현재 필터 조건에 해당하는 데이터가 없습니다. 필터 범위를 넓혀주세요.")
    st.stop()

# ============================================================
# KPI
# ============================================================
bicycle_mean = filtered["따릉이대여건수"].mean()
subway_mean = filtered["지하철승차인원"].mean()
temp_mean = filtered["기온(°C)"].mean()

bicycle_spearman = spearman_corr(
    filtered["기온(°C)"], filtered["따릉이대여건수"]
)
subway_spearman = spearman_corr(
    filtered["기온(°C)"], filtered["지하철승차인원"]
)


def kpi(col, label, value, help_text=""):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-help">{help_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


k1, k2, k3, k4, k5 = st.columns(5)
kpi(k1, "평균 기온", f"{temp_mean:.1f}°C", "선택 조건 내 평균")
kpi(k2, "평균 따릉이 대여", f"{bicycle_mean:,.0f}건", "시간대당 평균")
kpi(k3, "평균 지하철 승차", f"{subway_mean:,.0f}명", "시간대당 평균")
kpi(k4, "따릉이 Spearman", f"{bicycle_spearman:.3f}", "-1~+1 · 0에서 멀수록 관계가 강함")
kpi(k5, "지하철 Spearman", f"{subway_spearman:.3f}", "-1~+1 · 0에서 멀수록 관계가 강함")

# ============================================================
# 1. Hourly usage index
# ============================================================
st.markdown('<div class="section-title">시간대별 교통수단 이용 패턴</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">선택 조건에서 각 교통수단의 평균 이용량을 100으로 두고 시간대별 상대 이용 수준을 비교합니다.</div>',
    unsafe_allow_html=True,
)

hourly = calc_usage_index(filtered, "시간대")
hourly["시간대"] = pd.Categorical(
    hourly["시간대"], categories=TIME_ORDER, ordered=True
)
hourly = hourly.sort_values("시간대")

fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(
    hourly["시간대"], hourly["따릉이지수"],
    color=BICYCLE_COLOR, marker="o", linewidth=2.4, label="따릉이지수"
)
ax.plot(
    hourly["시간대"], hourly["지하철지수"],
    color=SUBWAY_COLOR, marker="o", linewidth=2.4, label="지하철지수"
)
ax.axhline(100, color="#777777", linestyle="--", linewidth=1.3, label="평균=100")
ax.set_xlabel("시간대")
ax.set_ylabel("평균=100 기준 이용지수")
ax.tick_params(axis="x", rotation=45)
ax.legend(ncol=3, frameon=False)
ax.grid(alpha=0.25)
fig.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

# ============================================================
# 2. Temperature vs usage
# ============================================================
st.markdown('<div class="section-title">기온과 이용량의 관계</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">산점도와 추세선을 통해 기온 변화와 이용량의 관계를 확인합니다. 상관관계는 인과관계를 의미하지 않습니다.</div>',
    unsafe_allow_html=True,
)

c1, c2 = st.columns(2)

with c1:
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    sns.regplot(
        data=filtered,
        x="기온(°C)",
        y="따릉이대여건수",
        scatter_kws={"alpha": 0.30, "s": 32, "color": BICYCLE_COLOR},
        line_kws={"color": "#6A1B9A", "linewidth": 2.2},
        ax=ax,
    )
    ax.set_title(
        f"기온 vs 따릉이 · Spearman {bicycle_spearman:.3f}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("기온 (°C)")
    ax.set_ylabel("따릉이 대여건수")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with c2:
    fig, ax = plt.subplots(figsize=(6.0, 4.3))
    sns.regplot(
        data=filtered,
        x="기온(°C)",
        y="지하철승차인원",
        scatter_kws={"alpha": 0.30, "s": 32, "color": SUBWAY_COLOR},
        line_kws={"color": "#6A1B9A", "linewidth": 2.2},
        ax=ax,
    )
    ax.set_title(
        f"기온 vs 지하철 · Spearman {subway_spearman:.3f}",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("기온 (°C)")
    ax.set_ylabel("지하철 승차인원")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

# ============================================================
# 3. Temperature-bin usage index
# ============================================================
st.markdown('<div class="section-title">기온 구간별 이용지수</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">전체 평균 이용량을 100으로 놓고, 각 기온 구간에서 평소보다 이용 수준이 높은지·낮은지를 비교합니다.</div>',
    unsafe_allow_html=True,
)

bin_edges = [-np.inf, 0, 5, 10, 15, 20, 25, 30, np.inf]
bin_labels = [
    "0°C 미만", "0~5°C", "5~10°C", "10~15°C",
    "15~20°C", "20~25°C", "25~30°C", "30°C 이상"
]

filtered["기온구간"] = pd.cut(
    filtered["기온(°C)"],
    bins=bin_edges,
    labels=bin_labels,
    right=False,
)

temp_index = calc_usage_index(filtered, "기온구간")
temp_index["기온구간"] = pd.Categorical(
    temp_index["기온구간"], categories=bin_labels, ordered=True
)
temp_index = temp_index.sort_values("기온구간")

fig, ax = plt.subplots(figsize=(12, 4.8))
ax.plot(
    temp_index["기온구간"], temp_index["따릉이지수"],
    color=BICYCLE_COLOR, marker="o", linewidth=2.4, label="따릉이지수"
)
ax.plot(
    temp_index["기온구간"], temp_index["지하철지수"],
    color=SUBWAY_COLOR, marker="o", linewidth=2.4, label="지하철지수"
)
ax.axhline(100, color="#777777", linestyle="--", linewidth=1.3, label="평균=100")
ax.set_xlabel("기온 구간")
ax.set_ylabel("평균=100 기준 이용지수")
ax.tick_params(axis="x", rotation=25)
ax.legend(frameon=False, ncol=3)
ax.grid(alpha=0.25)
fig.tight_layout()
st.pyplot(fig, width="stretch")
plt.close(fig)

# ============================================================
# 4. Month x hour heatmaps
# ============================================================
st.markdown('<div class="section-title">월 × 시간대 이용 패턴</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">선택한 월과 시간대에서 실제 평균 이용량의 계절·시간대 구조를 확인합니다.</div>',
    unsafe_allow_html=True,
)


def month_hour_pivot(data, value_col):
    temp = data.copy()
    temp["월표시"] = temp["월"].map(MONTH_LABEL)

    month_order = [
        MONTH_LABEL[m] for m in MONTH_ORDER if m in selected_months
    ]

    return (
        temp.pivot_table(
            index="월표시",
            columns="시간대",
            values=value_col,
            aggfunc="mean",
            observed=False,
        )
        .reindex(index=month_order, columns=TIME_ORDER)
    )


h1, h2 = st.columns(2)

with h1:
    pivot = month_hour_pivot(filtered, "따릉이대여건수")
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    sns.heatmap(
        pivot, annot=True, fmt=",.0f",
        cmap="Greens", linewidths=0.4,
        cbar=True, ax=ax
    )
    ax.set_title("월별·시간대별 평균 따릉이 대여건수", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("시간대")
    ax.set_ylabel("월")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with h2:
    pivot = month_hour_pivot(filtered, "지하철승차인원")
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    sns.heatmap(
        pivot, annot=True, fmt=",.0f",
        cmap="Blues", linewidths=0.4,
        cbar=True, ax=ax
    )
    ax.set_title("월별·시간대별 평균 지하철 승차인원", fontsize=12.5, fontweight="bold")
    ax.set_xlabel("시간대")
    ax.set_ylabel("월")
    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)

# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.caption(
    "※ 기본값은 평일·비공휴일입니다. 상관계수는 변수 간 관계의 정도와 방향을 나타내며 "
    "인과관계를 의미하지 않습니다."
)

if len(filtered) < 30:
    st.warning(
        f"현재 필터 조건의 관측값이 {len(filtered):,}개로 적습니다. "
        "Spearman 상관계수는 탐색적 참고용으로만 해석하세요."
    )
