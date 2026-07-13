import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import glob
import re
from collections import Counter
import os

# ============ 能力维度映射（用于雷达图） ============
DIMENSION_MAP = {
    # 技术硬实力
    'python': '技术硬实力', 'sql': '技术硬实力', 'hive': '技术硬实力',
    'spark': '技术硬实力', 'java': '技术硬实力', 'scala': '技术硬实力',
    'c++': '技术硬实力', 'c#': '技术硬实力', 'excel': '技术硬实力',
    'shell': '技术硬实力', 'linux': '技术硬实力', 'git': '技术硬实力',
    
    # 数据科学/建模
    'ab测试': '数据科学/建模', 'abtest': '数据科学/建模', 'a/b测试': '数据科学/建模',
    '机器学习': '数据科学/建模', '深度学习': '数据科学/建模', 'nlp': '数据科学/建模',
    '自然语言处理': '数据科学/建模', '回归': '数据科学/建模', '分类': '数据科学/建模',
    '聚类': '数据科学/建模', '预测': '数据科学/建模', '流失': '数据科学/建模',
    '用户画像': '数据科学/建模', '推荐算法': '数据科学/建模', '因果推断': '数据科学/建模',
    
    # 商业/游戏理解
    '留存': '商业/游戏理解', '付费': '商业/游戏理解', 'ltv': '商业/游戏理解',
    'dau': '商业/游戏理解', 'gmv': '商业/游戏理解', '归因': '商业/游戏理解',
    '增长': '商业/游戏理解', '用户运营': '商业/游戏理解', '活动分析': '商业/游戏理解',
    '商业化': '商业/游戏理解', '变现': '商业/游戏理解', 'roi': '商业/游戏理解',
    
    # 可视化/沟通
    'tableau': '可视化/沟通', 'power bi': '可视化/沟通', 'powerbi': '可视化/沟通',
    'finebi': '可视化/沟通', '可视化': '可视化/沟通', '看板': '可视化/沟通',
    '报告': '可视化/沟通', 'ppt': '可视化/沟通', '展示': '可视化/沟通',
    
    # 工程/工具
    'docker': '工程/工具', 'kubernetes': '工程/工具', 'k8s': '工程/工具',
    'airflow': '工程/工具', 'hadoop': '工程/工具', 'kafka': '工程/工具',
    'flink': '工程/工具', '数据仓库': '工程/工具', 'etl': '工程/工具',
}

# ============ 雷达图辅助函数 ============
def compute_market_dimension_scores(df, dim_map, dim_names):
    """计算市场上所有岗位的五维平均得分（百分比）"""
    dim_scores = {dim: [] for dim in dim_names}
    
    for idx, row in df.iterrows():
        text = str(row['jobDesc']).lower()
        # 每个岗位在每个维度只计一次（去重）
        hit_dims = set()
        for skill, dim in dim_map.items():
            if skill in text:
                hit_dims.add(dim)
        for dim in hit_dims:
            dim_scores[dim].append(1)
    
    # 计算每个维度的百分比（出现次数 / 总岗位数）
    result = {}
    total = len(df)
    for dim in dim_names:
        result[dim] = (len(dim_scores[dim]) / total * 100) if total > 0 else 0
    return result

def parse_resume_to_dimensions(resume_text, dim_map, dim_names):
    """把简历文本转化为五维得分（0-100分）"""
    dim_scores = {dim: 0 for dim in dim_names}
    text = resume_text.lower()
    
    for skill, dim in dim_map.items():
        if skill in text:
            dim_scores[dim] += 1
    
    # 归一化：每个技能 20 分，封顶 100
    for dim in dim_names:
        dim_scores[dim] = min(dim_scores[dim] * 20, 100)
    return dim_scores

def plot_radar_chart(market_scores, my_scores, dim_names):
    """绘制双雷达图（市场平均 vs 我的能力）"""
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # 我的能力（实线，醒目）
    my_values = [my_scores.get(d, 0) for d in dim_names]
    fig.add_trace(go.Scatterpolar(
        r=my_values,
        theta=dim_names,
        fill='toself',
        name='🎯 我的能力',
        line=dict(color='#2E86AB', width=3),
        fillcolor='rgba(46, 134, 171, 0.3)'
    ))
    
    # 市场需求平均（虚线，参考线）
    market_values = [market_scores.get(d, 0) for d in dim_names]
    fig.add_trace(go.Scatterpolar(
        r=market_values,
        theta=dim_names,
        fill='toself',
        name='📊 市场需求平均',
        line=dict(color='#E74C3C', width=2, dash='dash'),
        fillcolor='rgba(231, 76, 60, 0.1)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=12, weight='bold'))
        ),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=-0.15),
        height=450,
        margin=dict(l=40, r=40, t=20, b=40),
        title="能力画像雷达图（差距即学习方向）"
    )
    return fig

# 五维名称（固定顺序，保证雷达图一致）
DIM_NAMES = ['技术硬实力', '数据科学/建模', '商业/游戏理解', '可视化/沟通', '工程/工具']

# ============ 页面配置 ============
st.set_page_config(page_title="游戏行业岗位情报看板", layout="wide")
st.title("🎮 游戏行业数据分析岗位情报系统")

# ============ 数据加载 ============
@st.cache_data
def load_data():
    """自动加载 ./data 目录下所有 *_cleaned.csv 文件"""
    all_files = glob.glob("./data/*_cleaned.csv")
    if not all_files:
        st.error("❌ 未找到数据文件！请确保 ./data 目录下有 *_cleaned.csv 文件")
        return pd.DataFrame()
    
    dfs = []
    for f in all_files:
        df = pd.read_csv(f, encoding='utf-8-sig')
        # 从文件名提取公司名
        company = os.path.basename(f).replace("_cleaned.csv", "").replace("recruits_", "")
        df['company'] = company
        dfs.append(df)
    
    df = pd.concat(dfs, ignore_index=True)
    
    # 清洗薪资字段（如果存在）
    if 'salary_mid' in df.columns:
        df['salary_mid'] = pd.to_numeric(df['salary_mid'], errors='coerce')
    
    return df

df = load_data()

if df.empty:
    st.stop()

# ============ 侧边栏筛选器 ============
st.sidebar.header("🔍 筛选条件")

# 公司筛选
companies = ['全部'] + sorted(df['company'].unique().tolist())
selected_company = st.sidebar.selectbox("选择公司", companies)

# 城市筛选
cities = ['全部'] + sorted(df['city'].dropna().unique().tolist())
selected_city = st.sidebar.selectbox("选择城市", cities)

# 薪资范围筛选
min_salary = int(df['salary_mid'].min()) if not df['salary_mid'].isna().all() else 0
max_salary = int(df['salary_mid'].max()) if not df['salary_mid'].isna().all() else 100
salary_range = st.sidebar.slider("薪资范围 (K)", min_salary, max_salary, (min_salary, max_salary))

# 应用筛选
filtered_df = df.copy()
if selected_company != '全部':
    filtered_df = filtered_df[filtered_df['company'] == selected_company]
if selected_city != '全部':
    filtered_df = filtered_df[filtered_df['city'] == selected_city]
filtered_df = filtered_df[(filtered_df['salary_mid'] >= salary_range[0]) & 
                          (filtered_df['salary_mid'] <= salary_range[1])]

# ============ 顶部 KPI 卡片 ============
st.subheader("📊 市场概览")
col1, col2, col3, col4 = st.columns(4)
col1.metric("📌 岗位总数", len(filtered_df))
col2.metric("💰 薪资中位数", f"{filtered_df['salary_mid'].median():.1f}K" if not filtered_df.empty else "N/A")
col3.metric("🎓 平均经验要求", f"{filtered_df['exp_min'].mean():.1f}年" if 'exp_min' in filtered_df.columns else "N/A")
col4.metric("🏢 公司数", filtered_df['company'].nunique())

st.divider()

# ============ 第一行：薪资分布 + 公司对比 ============
col1, col2 = st.columns(2)

with col1:
    st.subheader("💰 薪资分布")
    if not filtered_df.empty and 'salary_mid' in filtered_df.columns:
        fig = px.histogram(
            filtered_df, 
            x='salary_mid',
            nbins=30,
            title="岗位薪资分布（K/月）",
            labels={'salary_mid': '薪资 (K)', 'count': '岗位数'},
            color_discrete_sequence=['#4CAF50']
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🏢 各公司薪资对比")
    if not filtered_df.empty and 'company' in filtered_df.columns and 'salary_mid' in filtered_df.columns:
        company_stats = filtered_df.groupby('company')['salary_mid'].median().sort_values(ascending=False).reset_index()
        fig = px.bar(
            company_stats,
            x='company',
            y='salary_mid',
            title="各公司薪资中位数对比",
            labels={'company': '公司', 'salary_mid': '薪资中位数 (K)'},
            color='salary_mid',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============ 第二行：技能排名 + 城市分布 ============
col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 高频技能排名")
    # 从 jobDesc 中提取技能关键词（使用你的技能词库）
    # 这里先动态提取高频词，你可以替换成读取 skills.txt
    if 'jobDesc' in filtered_df.columns and not filtered_df.empty:
        all_text = ' '.join(filtered_df['jobDesc'].dropna().astype(str))
        # 提取英文技能词（简单版本）
        eng_words = re.findall(r'[A-Za-z][A-Za-z0-9\+\#\.\-]+[A-Za-z]', all_text)
        # 过滤常见停用词
        stopwords = {'and', 'the', 'for', 'with', 'are', 'you', 'will', 'your', 'this', 'from', 'have', 'not', 'can', 'ability'}
        word_counts = Counter([w.lower() for w in eng_words if len(w) > 2 and w.lower() not in stopwords])
        top_skills = word_counts.most_common(15)
        
        if top_skills:
            skill_df = pd.DataFrame(top_skills, columns=['技能', '出现次数'])
            fig = px.bar(
                skill_df,
                x='出现次数',
                y='技能',
                orientation='h',
                title="Top 15 高频技能",
                color='出现次数',
                color_continuous_scale='Oranges'
            )
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无技能数据")

with col2:
    st.subheader("📍 城市分布")
    if not filtered_df.empty and 'city' in filtered_df.columns:
        city_counts = filtered_df['city'].value_counts().reset_index()
        city_counts.columns = ['城市', '岗位数']
        fig = px.pie(
            city_counts.head(8),
            values='岗位数',
            names='城市',
            title="Top 8 招聘城市",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============ 第三行：学历要求 ============
st.subheader("🎓 学历要求分布")
if not filtered_df.empty and 'education' in filtered_df.columns:
    edu_counts = filtered_df['education'].value_counts().reset_index()
    edu_counts.columns = ['学历', '岗位数']
    fig = px.bar(
        edu_counts,
        x='学历',
        y='岗位数',
        title="学历要求分布",
        color='岗位数',
        color_continuous_scale='Purples'
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

# ============ 招聘趋势分析（按年分界，修正版） ============
st.subheader("📈 招聘趋势与热度预测")

if not filtered_df.empty and 'releaseDate' in filtered_df.columns:
    # 准备数据
    temp_df = filtered_df.copy()
    temp_df['date'] = pd.to_datetime(temp_df['releaseDate'], errors='coerce')
    temp_df['year'] = temp_df['date'].dt.year
    temp_df['month'] = temp_df['date'].dt.to_period('M').astype(str)
    
    # 按月聚合
    trend = temp_df.groupby('month').size().reset_index(name='count')
    trend = trend.sort_values('month')
    
    if not trend.empty and len(trend) > 1:
        # 增加数值索引（用于坐标轴）
        trend['month_idx'] = range(len(trend))
        
        # 创建 Figure
        fig = go.Figure()
        
        # 添加主趋势线（使用 month_idx 作为 x）
        fig.add_trace(go.Scatter(
            x=trend['month_idx'],
            y=trend['count'],
            mode='lines+markers',
            name='岗位数量',
            line=dict(color='#2E86AB', width=2.5),
            marker=dict(size=6, color='#2E86AB'),
            hovertemplate='月份: %{customdata}<br>岗位数: %{y}<extra></extra>',
            customdata=trend['month']
        ))
        
        # 自定义 x 轴刻度标签（显示月份字符串）
        tick_vals = trend['month_idx'].tolist()
        tick_text = trend['month'].tolist()
        
        # --- 添加年份分界线 ---
        years = sorted(temp_df['year'].dropna().unique())
        # 获取每个年份的第一个月份的索引
        year_first_idx = {}
        for year in years:
            first_month = temp_df[temp_df['year'] == year]['month'].min()
            if first_month:
                # 找到该月份对应的索引
                idx = trend[trend['month'] == first_month]['month_idx'].values
                if len(idx) > 0:
                    year_first_idx[year] = idx[0]
        
        # 添加垂直虚线分界线
        for year, idx in year_first_idx.items():
            if year == years[0]:
                continue  # 跳过第一年，不需要在开头画线
            # 在当年第一个月份位置画线
            fig.add_vline(
                x=idx,
                line_width=1.5,
                line_dash="dash",
                line_color="#E74C3C",
                opacity=0.7
            )
            # 添加年份注释
            fig.add_annotation(
                x=idx,
                y=trend['count'].max() * 0.95,
                text=f"📅 {year}",
                showarrow=False,
                font=dict(color="#E74C3C", size=13, weight="bold"),
                xanchor="right",
                yanchor="top"
            )
        
        # 添加年度背景色块（使用 vrect）
        colors = ['rgba(46, 134, 171, 0.08)', 'rgba(231, 76, 60, 0.08)',
                  'rgba(46, 204, 113, 0.08)', 'rgba(241, 196, 15, 0.08)']
        for i, year in enumerate(years):
            year_data = temp_df[temp_df['year'] == year]
            if not year_data.empty:
                # 获取该年第一个和最后一个月份的索引
                first_month = year_data['month'].min()
                last_month = year_data['month'].max()
                start_idx = trend[trend['month'] == first_month]['month_idx'].values
                end_idx = trend[trend['month'] == last_month]['month_idx'].values
                if len(start_idx) > 0 and len(end_idx) > 0:
                    start = start_idx[0]
                    end = end_idx[0]
                    # 如果是最后一年，延长到数据末尾
                    if year == years[-1]:
                        end = trend['month_idx'].max()
                    fig.add_vrect(
                        x0=start,
                        x1=end,
                        fillcolor=colors[i % len(colors)],
                        opacity=0.3,
                        layer="below",
                        line_width=0
                    )
        
        # 更新布局
        fig.update_layout(
            title="月度岗位发布趋势（按年分界）",
            xaxis_title="月份",
            yaxis_title="岗位数量",
            height=420,
            hovermode="x unified",
            showlegend=False,
            xaxis=dict(
                tickvals=tick_vals[::2],  # 每两个月显示一个标签，防止拥挤
                ticktext=[tick_text[i] for i in range(0, len(tick_text), 2)],
                tickangle=45,
                tickfont=dict(size=11)
            ),
            yaxis=dict(
                gridcolor='lightgray',
                gridwidth=0.5
            ),
            margin=dict(l=20, r=20, t=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ===== 热度预测（不变） =====
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        # 年同比计算
        years_list = sorted(temp_df['year'].dropna().unique())
        if len(years_list) >= 2:
            year_counts = temp_df.groupby('year').size()
            current_year = years_list[-1]
            prev_year = years_list[-2]
            current_count = year_counts[current_year] if current_year in year_counts else 0
            prev_count = year_counts[prev_year] if prev_year in year_counts else 0
            if prev_count > 0:
                yoy_change = ((current_count - prev_count) / prev_count * 100)
                col1.metric(f"📊 {current_year}年岗位数", f"{current_count}个",
                            delta=f"{yoy_change:.1f}%", delta_color="normal")
            else:
                col1.metric(f"📊 {current_year}年岗位数", f"{current_count}个")
        else:
            col1.metric("📊 总岗位数", f"{len(temp_df)}个")
        
        if len(trend) >= 3:
            recent_trend = trend.tail(3)['count'].tolist()
            if recent_trend[-1] > recent_trend[0]:
                col2.metric("🔥 当前热度", "上升 🔼", delta="岗位在增加")
            elif recent_trend[-1] < recent_trend[0]:
                col2.metric("🔥 当前热度", "下降 🔽", delta="岗位在减少")
            else:
                col2.metric("🔥 当前热度", "平稳 ➡️", delta="与之前持平")
        else:
            col2.metric("🔥 当前热度", "数据不足", delta="需要更多数据")
        
        if not temp_df.empty:
            month_counts = temp_df.groupby(temp_df['date'].dt.month).size()
            if not month_counts.empty:
                peak_month = month_counts.idxmax()
                month_names = {1:'1月',2:'2月',3:'3月',4:'4月',5:'5月',6:'6月',
                               7:'7月',8:'8月',9:'9月',10:'10月',11:'11月',12:'12月'}
                col3.metric("📌 招聘旺季", month_names.get(peak_month, f"{peak_month}月"),
                           delta=f"{month_counts.max()}个岗位")
        
        if len(trend) >= 2:
            last = trend.iloc[-1]
            prev = trend.iloc[-2]
            diff = last['count'] - prev['count']
            col4.metric("📅 最近月份", last['month'],
                       delta=f"{'+' if diff >= 0 else ''}{diff}个", delta_color="normal")
        
    else:
        st.info("📊 数据不足，需要至少 2 个时间点的数据才能显示趋势")
else:
    st.info("📊 当前数据中没有 'releaseDate' 字段，无法分析趋势")

# ============ 简历匹配模块 ============
st.subheader("🎯 简历匹配器")
st.markdown("粘贴你的简历文本，系统会自动计算与所有岗位的匹配度，并推荐最适合你的岗位。")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 预计算所有 JD 的向量（放在 @st.cache_data 里避免重复计算）
@st.cache_data
def compute_jd_vectors(texts):
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    vectors = tfidf.fit_transform(texts)
    return tfidf, vectors

# 获取 JD 文本（填充空值）
jd_texts = df['jobDesc'].fillna('').tolist()
tfidf, jd_vectors = compute_jd_vectors(jd_texts)

# 输入框（用户粘贴简历）
your_text = st.text_area("📄 请粘贴你的简历文本（至少 50 字）", height=200)

if your_text and len(your_text) > 20:
    # 向量化简历
    resume_vector = tfidf.transform([your_text])
    # 计算相似度
    similarities = cosine_similarity(resume_vector, jd_vectors).flatten()
    df['match_score'] = similarities
    
    # 展示 Top 10
    top_matches = df.sort_values('match_score', ascending=False)[['jobName', 'company', 'city', 'salary_mid', 'match_score']].head(10)
    top_matches['match_score'] = top_matches['match_score'].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(top_matches, use_container_width=True)


    # ========== 新增：雷达图 + 差距分析 ==========
    st.markdown("---")
    st.subheader("📊 能力画像对比与差距分析")
        
     # 1. 计算市场平均五维得分（缓存计算结果）
    market_scores = compute_market_dimension_scores(df, DIMENSION_MAP, DIM_NAMES)
    
    # 2. 解析简历，得到我的五维得分
    my_scores = parse_resume_to_dimensions(your_text, DIMENSION_MAP, DIM_NAMES)
    
    # 3. 画雷达图
    fig = plot_radar_chart(market_scores, my_scores, DIM_NAMES)
    st.plotly_chart(fig, use_container_width=True)
    
    # 4. 差距分析列表
    col1, col2 = st.columns(2)
    
    # 计算差距
    strengths = []
    blind_spots = []
    for dim in DIM_NAMES:
        gap = market_scores[dim] - my_scores[dim]
        if gap < -5:  # 我超出市场 5% 以上
            strengths.append((dim, abs(gap)))
        elif gap > 5:  # 我落后市场 5% 以上
            blind_spots.append((dim, gap))
    
    with col1:
        if strengths:
            st.markdown("**✅ 你的优势维度（继续保持）**")
            for dim, gap in sorted(strengths, key=lambda x: x[1], reverse=True):
                st.success(f"**{dim}**：超出市场 {gap:.0f}%")
        else:
            st.info("暂无显著优势维度")
    
    with col2:
        if blind_spots:
            st.markdown("**🔴 需要重点补强的维度（面试前突击！）**")
            for dim, gap in sorted(blind_spots, key=lambda x: x[1], reverse=True):
                st.error(f"**{dim}**：低于市场 {gap:.0f}%")
        else:
            st.info("🎉 你的能力覆盖全面，没有明显短板！")
    
    # 5. 针对性学习建议（简单版，不依赖 AI）
    if blind_spots:
        st.markdown("---")
        st.markdown("**💡 针对薄弱维度的学习建议**")
        tips = {
            '技术硬实力': '👉 加强 SQL 和 Python 练习，推荐 LeetCode + 牛客网 SQL 题库',
            '数据科学/建模': '👉 学习 A/B 测试理论基础（如《关键迭代》），掌握 sklearn 建模流程',
            '商业/游戏理解': '👉 关注游戏行业报告（如伽马数据），理解 LTV、留存、DAU 等核心指标',
            '可视化/沟通': '👉 练习 Tableau/Power BI 制作看板，准备 2-3 个数据故事',
            '工程/工具': '👉 学习 Git 基本操作 + Docker 入门，了解数据仓库基础概念'
        }
        for dim, _ in blind_spots[:2]:  # 只显示前两个最弱的
            if dim in tips:
                st.write(tips[dim])
    
    # 额外推荐：显示最缺的 3 个技能（从最匹配的 JD 中提取高频词）
    top_jd = df.sort_values('match_score', ascending=False).head(5)['jobDesc'].fillna('')
    all_text = ' '.join(top_jd)
    # 简单提取高频英文技能词
    import re
    from collections import Counter
    words = re.findall(r'[A-Za-z][A-Za-z0-9\+#\.\-]+[A-Za-z]', all_text)
    common_skills = Counter([w.lower() for w in words if len(w) > 2]).most_common(5)
    if common_skills:
        st.markdown("**💡 建议重点学习以下技能（来自最匹配岗位）**：")
        st.write(", ".join([f"`{skill}`" for skill, _ in common_skills]))
else:
    st.info("请输入简历文本（至少 20 字）以获取匹配结果。")

# ============ 底部 ============
st.divider()
st.caption(f"📊 数据来源：招聘平台 | 共 {len(filtered_df)} 条记录 | 最后更新：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")