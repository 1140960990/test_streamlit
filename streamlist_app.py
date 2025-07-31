import py_compile
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# 设置页面配置
st.set_page_config(
    page_title="负荷预测算法平台",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        background-color: #1f4e79;
        color: white;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f4e79;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 生成模拟数据
def generate_comprehensive_data():
    # 生成24小时的时间序列
    times = pd.date_range("2024-09-03 00:00", "2024-09-03 23:00", freq="H")
    
    # 生成实际负荷数据（更真实的模式）
    base_load = 800  # 基础负荷
    peak_hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]  # 峰时段
    valley_hours = [1, 2, 3, 4, 5, 6]  # 谷时段
    
    actual_loads = []
    for hour in range(24):
        if hour in peak_hours:
            load = base_load + np.random.normal(200, 30)  # 峰时段
        elif hour in valley_hours:
            load = base_load + np.random.normal(-100, 20)  # 谷时段
        else:
            load = base_load + np.random.normal(50, 15)   # 平时段
        actual_loads.append(max(load, 600))  # 确保最小负荷
    
    # 生成不同算法的预测数据
    algorithms = {
        "时序算法": {"bias": 0, "noise": 0.02},
        "分段时序算法": {"bias": -0.01, "noise": 0.015},
        "回归算法": {"bias": 0.005, "noise": 0.025},
        "GBDT算法": {"bias": -0.002, "noise": 0.018}
    }
    
    data = {"时间": times, "实际负荷": actual_loads}
    
    for algo_name, params in algorithms.items():
        predicted_loads = []
        for actual in actual_loads:
            # 添加偏差和噪声
            prediction = actual * (1 + params["bias"] + np.random.normal(0, params["noise"]))
            predicted_loads.append(prediction)
        data[f"{algo_name}_预测"] = predicted_loads
    
    return pd.DataFrame(data)

# 计算准确率
def calculate_accuracy(actual, predicted):
    return max(0, 100 - abs((predicted - actual) / actual * 100))

# 生成数据
df = generate_comprehensive_data()

# 主标题
st.markdown('<div class="main-header"><h1>⚡ 负荷预测算法平台</h1></div>', unsafe_allow_html=True)

# 配置面板
col1, col2, col3, col4 = st.columns(4)

with col1:
    prediction_type = st.selectbox(
        "预测类型",
        ["营销口径负荷预测","公司级负荷预测",  "用户负荷预测", "新能源发电预测"],
        index=0
    )

with col2:
    prediction_dimension = st.selectbox(
        "预测维度",
        ["短期", "中期", "长期"],
        index=0
    )

with col3:
    date_range = st.date_input(
        "查询时间",
        value=(pd.to_datetime("2024-09-03"), pd.to_datetime("2024-09-03")),
        max_value=pd.to_datetime("2024-12-31")
    )

with col4:
    st.write("")
    col4_1, col4_2, col4_3 = st.columns(3)
    with col4_1:
        query_btn = st.button("查询", type="primary")
    with col4_2:
        reset_btn = st.button("重置")
    with col4_3:
        export_btn = st.button("导出预测结果")

# 算法选择和基准线设置
col5, col6 = st.columns([2, 1])

with col5:
    st.markdown("**算法选择**")
    algorithms = ["时序算法", "分段时序算法", "回归算法", "GBDT算法"]
    selected_algorithms = []
    cols = st.columns(4)
    for i, algo in enumerate(algorithms):
        with cols[i]:
            if st.checkbox(algo, value=(i==0)):
                selected_algorithms.append(algo)

with col6:
    st.markdown("**基准线设置**")
    baseline = st.number_input("准确率基准线 (%)", min_value=0, max_value=100, value=95)

# 创建两个标签页
tab1, tab2 = st.tabs(["📊 负荷曲线对比", "📈 准确率曲线"])

with tab1:
    st.markdown("### 预测负荷曲线")
    
    if selected_algorithms:
        # 创建负荷对比图表
        fig_load = go.Figure()
        
        # 添加实际负荷线
        fig_load.add_trace(go.Scatter(
            x=df['时间'],
            y=df['实际负荷'],
            mode='lines+markers',
            name='实际负荷',
            line=dict(color='black', width=3),
            marker=dict(size=6)
        ))
        
        # 添加预测负荷线
        colors = ['blue', 'red', 'green', 'orange']
        for i, algo in enumerate(selected_algorithms):
            if f"{algo}_预测" in df.columns:
                fig_load.add_trace(go.Scatter(
                    x=df['时间'],
                    y=df[f"{algo}_预测"],
                    mode='lines+markers',
                    name=f'{algo}预测',
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4)
                ))
        
        # 设置图表布局
        fig_load.update_layout(
            title="负荷预测对比图",
            xaxis_title="时间",
            yaxis_title="负荷 (MW)",
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_load, use_container_width=True)
        
        # # 时间段背景色说明
        # st.markdown("**时间段说明：**")
        # col_legend = st.columns(4)
        # with col_legend[0]:
        #     st.markdown("🔴 尖时段")
        # with col_legend[1]:
        #     st.markdown("🟠 峰时段")
        # with col_legend[2]:
        #     st.markdown("🟡 平时段")
        # with col_legend[3]:
        #     st.markdown("🟢 谷时段")
        
        # 负荷数据表格（横向按时间排列，实际负荷和预测负荷分两行展示）
        st.markdown("### 负荷预测数据详情")

        # 先准备表头
        table_columns = [
            "序号", 
            "预测算法", 
            "预测最大负荷", 
            "最大负荷时刻", 
            "预测最小负荷", 
            "最小负荷时刻"
        ]
        time_list = df['时间'].dt.strftime("%H:%M").tolist()
        table_columns.extend(time_list)

        table_data = []
        seq = 1
        for algo in selected_algorithms:
            if f"{algo}_预测" in df.columns:
                # 实际负荷行
                actual_row = [seq, f"{algo} 实际", "", "", "", ""]
                actual_row.extend([f"{v:.2f}" for v in df['实际负荷']])
                table_data.append(actual_row)
                # 预测负荷行
                pred_values = df[f"{algo}_预测"]
                max_pred = pred_values.max()
                min_pred = pred_values.min()
                max_time = df.loc[pred_values.idxmax(), '时间'].strftime("%H:%M")
                min_time = df.loc[pred_values.idxmin(), '时间'].strftime("%H:%M")
                pred_row = [
                    "", 
                    f"{algo} 预测", 
                    f"{max_pred:.2f}", 
                    max_time, 
                    f"{min_pred:.2f}", 
                    min_time
                ]
                pred_row.extend([f"{v:.2f}" for v in pred_values])
                table_data.append(pred_row)
                seq += 1

        # 构建DataFrame
        load_table_df = pd.DataFrame(table_data, columns=table_columns)
        st.dataframe(load_table_df, use_container_width=True)
        
        # # 负荷数据表格（横向按时间排列，实际负荷和预测负荷分两行展示）
        # st.markdown("### 负荷预测数据详情")

        # table_columns = ["序号", "预测算法"]
        # time_list = df['时间'].dt.strftime("%H:%M").tolist()
        # table_columns.extend(time_list)

        # table_data = []
        # seq = 1
        # for algo in selected_algorithms:
        #     if f"{algo}_预测" in df.columns:
        #         # 实际负荷行
        #         actual_row = [seq, f"{algo} 实际"]
        #         actual_row.extend([f"{v:.2f}" for v in df['实际负荷']])
        #         table_data.append(actual_row)
        #         # 预测负荷行
        #         pred_row = ["", f"{algo} 预测"]
        #         pred_row.extend([f"{v:.2f}" for v in df[f"{algo}_预测"]])
        #         table_data.append(pred_row)
        #         seq += 1

        # # 构建DataFrame
        # load_table_df = pd.DataFrame(table_data, columns=table_columns)
        # st.dataframe(load_table_df, use_container_width=True)

with tab2:
    st.markdown("### 预测准确率分析")
    
    if selected_algorithms:
        # 创建准确率图表
        fig_accuracy = go.Figure()
        
        # 添加准确率曲线
        colors = ['blue', 'red', 'green', 'orange']
        for i, algo in enumerate(selected_algorithms):
            if f"{algo}_预测" in df.columns:
                accuracies = [calculate_accuracy(actual, pred) 
                            for actual, pred in zip(df['实际负荷'], df[f"{algo}_预测"])]
                fig_accuracy.add_trace(go.Scatter(
                    x=df['时间'],
                    y=accuracies,
                    mode='lines+markers',
                    name=f'{algo}准确率',
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=4)
                ))
        
        # 添加基准线
        fig_accuracy.add_hline(y=baseline, line_dash="dash", line_color="red", 
                             annotation_text=f"基准线 {baseline}%")
        
        # 设置图表布局
        fig_accuracy.update_layout(
            title="预测准确率曲线",
            xaxis_title="时间",
            yaxis_title="准确率 (%)",
            yaxis=dict(range=[0, 100]),
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig_accuracy, use_container_width=True)
        
        # 准确率统计指标
        st.markdown("### 算法性能统计")
        col_stats = st.columns(len(selected_algorithms))
        
        for i, algo in enumerate(selected_algorithms):
            if f"{algo}_预测" in df.columns:
                with col_stats[i]:
                    accuracies = [calculate_accuracy(actual, pred) 
                                for actual, pred in zip(df['实际负荷'], df[f"{algo}_预测"])]
                    avg_accuracy = np.mean(accuracies)
                    max_accuracy = np.max(accuracies)
                    min_accuracy = np.min(accuracies)
                    std_accuracy = np.std(accuracies)
                    
                    st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"**{algo}**")
                    st.metric("平均准确率", f"{avg_accuracy:.2f}%")
                    st.metric("最高准确率", f"{max_accuracy:.2f}%")
                    st.metric("最低准确率", f"{min_accuracy:.2f}%")
                    st.metric("标准差", f"{std_accuracy:.2f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # 按时间横排展示，分三行：实际负荷、预测负荷、准确率
        st.markdown("### 准确率数据详情")
        for algo in selected_algorithms:
            if f"{algo}_预测" in df.columns:
                st.markdown(f"#### 算法：{algo}")
                times = [row.strftime("%H:%M") for row in df['时间']]
                actual_loads = [f"{v:.2f}" for v in df['实际负荷']]
                predicted_loads = [f"{v:.2f}" for v in df[f"{algo}_预测"]]
                accuracies = [f"{calculate_accuracy(a, p):.2f}%" for a, p in zip(df['实际负荷'], df[f"{algo}_预测"])]
                meets_baseline = [calculate_accuracy(a, p) >= baseline for a, p in zip(df['实际负荷'], df[f"{algo}_预测"])]
                # 构建三行数据
                actual_row = ["实际负荷"] + actual_loads
                predicted_row = ["预测负荷"] + predicted_loads
                accuracy_row = ["准确率"] + [
                    f"{acc} {'✅' if meet else '❌'}" for acc, meet in zip(accuracies, meets_baseline)
                ]
                # 构建表格
                table_data = [actual_row, predicted_row, accuracy_row]
                # 构建列名
                columns = [""] + times
                # 转为DataFrame
                table_df = pd.DataFrame(table_data, columns=columns)
                st.dataframe(table_df, use_container_width=True)


# 侧边栏
with st.sidebar:
    st.markdown("### 🔧 系统设置")
    
    # 用户信息
    st.markdown("**当前用户：** XXX")
    
    # 系统参数
    st.markdown("**系统参数**")
    refresh_rate = st.selectbox("数据刷新频率", ["5分钟", "10分钟", "30分钟", "1小时"], index=1)
    chart_theme = st.selectbox("图表主题", ["默认", "深色", "浅色"], index=0)
    
    # 通知设置
    st.markdown("**通知设置**")
    st.checkbox("预测偏差告警", value=True)
    st.checkbox("系统状态通知", value=True)
    st.checkbox("数据更新提醒", value=False)
    
    # 快捷操作
    st.markdown("### ⚡ 快捷操作")
    if st.button("刷新数据", type="primary"):
        st.rerun()
    
    if st.button("导出报告"):
        st.success("报告导出成功！")
    
    if st.button("系统诊断"):
        st.info("系统运行正常")

# 页脚
st.markdown("---")
