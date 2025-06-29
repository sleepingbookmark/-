import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import (classification_report, mean_squared_error,
                           confusion_matrix, roc_curve, auc, silhouette_score)
import joblib
import io

# 页面配置
st.set_page_config(page_title="数据分析平台", layout="wide")

# 标题
st.title("数据分析与机器学习平台")

# 侧边栏菜单
menu = st.sidebar.selectbox("功能菜单",
    ["数据导入", "数据预览", "数据处理", "数据分析", "数据建模"])

# 全局变量存储数据
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.original_df = None
if 'model' not in st.session_state:
    st.session_state.model = None

# 数据导入模块
if menu == "数据导入":
    st.header("数据导入")
    uploaded_file = st.file_uploader("上传CSV或Excel文件", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            st.session_state.original_df = df.copy()
            st.success("数据导入成功！")
            
            st.success("数据导入成功！")
            
        except Exception as e:
            st.error(f"导入失败: {e}")

# 数据处理模块
elif menu == "数据处理" and st.session_state.df is not None:
    st.header("数据处理")
    df = st.session_state.df
    
    # 检查重复值
    st.subheader("重复值检查")
    dup_rows = df.duplicated().sum()
    st.write(f"发现 {dup_rows} 个重复行")
    if dup_rows > 0:
        if st.button("删除重复值"):
            df = df.drop_duplicates()
            st.session_state.df = df
            st.success(f"已删除 {dup_rows} 个重复行")
    
    # 缺失值处理
    st.subheader("缺失值处理")
    missing = df.isnull().sum()
    st.write("各列缺失值数量:")
    st.dataframe(missing[missing > 0])
    
    if missing.sum() > 0:
        col_option = st.selectbox("选择要处理的列", missing[missing > 0].index)
        method = st.radio("处理方法", 
            ["删除行", "填充均值", "填充中位数", "填充众数", "自定义值"])
        
        if st.button("应用处理"):
            if method == "删除行":
                df = df.dropna(subset=[col_option])
            else:
                if method == "填充均值":
                    fill_value = df[col_option].mean()
                elif method == "填充中位数":
                    fill_value = df[col_option].median()
                elif method == "填充众数":
                    fill_value = df[col_option].mode()[0]
                else:
                    fill_value = st.text_input("输入填充值")
                
                df[col_option] = df[col_option].fillna(fill_value)
            
            st.session_state.df = df
            st.success("缺失值处理完成")
    
    # 数据编辑
    st.subheader("数据编辑")
    edit_col = st.selectbox("选择要编辑的列", df.columns)
    
    # 条件选择
    st.subheader("选择替换条件")
    condition_type = st.radio("条件类型",
        ["特定值", "数值范围", "包含文本"])
    
    if condition_type == "特定值":
        old_value = st.text_input("输入要替换的值")
    elif condition_type == "数值范围":
        if df[edit_col].dtype in ['int64', 'float64']:
            min_val = st.number_input("最小值", value=float(df[edit_col].min()))
            max_val = st.number_input("最大值", value=float(df[edit_col].max()))
        else:
            st.warning("该列不是数值类型，不能使用范围条件")
    else:  # 包含文本
        old_value = st.text_input("输入包含的文本")
    
    # 新值输入
    new_value = st.text_input("输入新值")
    
    if st.button("执行替换"):
        if condition_type == "特定值":
            df[edit_col] = df[edit_col].replace(old_value, new_value)
        elif condition_type == "数值范围" and df[edit_col].dtype in ['int64', 'float64']:
            mask = (df[edit_col] >= min_val) & (df[edit_col] <= max_val)
            df.loc[mask, edit_col] = new_value
        elif condition_type == "包含文本":
            df[edit_col] = df[edit_col].astype(str).str.replace(old_value, new_value)
        
        st.session_state.df = df
        st.success("值已更新")
    
    # 离群值检测
    st.subheader("离群值检测")
    outlier_col = st.selectbox("选择检测列", df.select_dtypes(include=['int64','float64']).columns)
    
    if outlier_col:
        # 箱型图可视化
        fig, ax = plt.subplots()
        sns.boxplot(x=df[outlier_col], ax=ax)
        st.pyplot(fig)
        
        method = st.selectbox("检测方法",
            ["IQR方法", "Z-score方法", "百分位法"])
        
        if method == "IQR方法":
            Q1 = df[outlier_col].quantile(0.25)
            Q3 = df[outlier_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        elif method == "Z-score方法":
            mean = df[outlier_col].mean()
            std = df[outlier_col].std()
            lower_bound = mean - 3*std
            upper_bound = mean + 3*std
        else:  # 百分位法
            lower_bound = df[outlier_col].quantile(0.01)
            upper_bound = df[outlier_col].quantile(0.99)
        
        outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
        st.write(f"检测到 {len(outliers)} 个离群值")
        st.dataframe(outliers)
        
        if len(outliers) > 0:
            action = st.radio("处理方式",
                ["替换为边界值", "删除离群值所在行"])
            
            if st.button("应用处理"):
                if action == "替换为边界值":
                    df[outlier_col] = df[outlier_col].clip(lower_bound, upper_bound)
                else:
                    df = df[(df[outlier_col] >= lower_bound) & (df[outlier_col] <= upper_bound)]
                
                st.session_state.df = df
                st.success(f"已处理 {len(outliers)} 个离群值")
                
        # 导出处理后的数据
        if st.button("导出清洗后数据"):
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下载CSV",
                data=csv,
                file_name="cleaned_data.csv",
                mime='text/csv'
            )

# 数据分析模块
elif menu == "数据分析" and st.session_state.df is not None:
    st.header("数据分析")
    df = st.session_state.df
    
    # 单变量分析
    st.subheader("单变量分析")
    analysis_col = st.selectbox("选择分析变量", df.columns)
    plot_type = st.selectbox("选择图表类型",
                           ["直方图", "箱线图", "饼图", "密度图", "计数图"])
    
    # 直方图特殊设置
    if plot_type == "直方图":
        show_kde = st.checkbox("显示拟合曲线", value=True)
    elif plot_type == "密度图":
        show_shade = st.checkbox("显示曲线阴影", value=True)
    
    # 坐标轴设置
    with st.expander("坐标轴设置"):
        # 比例调节
        x_scale = st.selectbox("X轴比例", ["linear", "log", "symlog"])
        y_scale = st.selectbox("Y轴比例", ["linear", "log", "symlog"])
        
        # 刻度设置
        tick_rotation = st.slider("X轴刻度旋转角度", 0, 90, 30)
        tick_fontsize = st.slider("刻度字体大小", 6, 14, 10)
        
        # 范围调节
        if df[analysis_col].dtype in ['int64', 'float64']:
            min_val, max_val = float(df[analysis_col].min()), float(df[analysis_col].max())
            y_range = st.slider("Y轴范围", min_val, max_val, (min_val, max_val))
    
    # 生成图像按钮
    if st.button("生成图像"):
    
        if df[analysis_col].dtype in ['int64', 'float64']:
            # 数值型变量
            fig, ax = plt.subplots()
            
            if plot_type == "直方图":
                sns.histplot(x=df[analysis_col], kde=True, ax=ax)
                ax.set_xlabel(analysis_col)
                ax.set_ylabel("频数")
                plt.xticks(rotation=tick_rotation, ha="right", fontsize=tick_fontsize)
                plt.tight_layout()
            elif plot_type == "箱线图":
                sns.boxplot(x=df[analysis_col], ax=ax)
                ax.set_ylabel("数值")
                ax.set_xlabel(analysis_col)
                plt.xticks(rotation=tick_rotation, ha="right", fontsize=tick_fontsize)
                plt.tight_layout()
            elif plot_type == "密度图":
                sns.kdeplot(x=df[analysis_col], ax=ax, shade=show_shade)
                ax.set_ylabel(analysis_col)
                ax.set_xlabel("密度")
                plt.xticks(rotation=tick_rotation, ha="right", fontsize=tick_fontsize)
                plt.tight_layout()
            elif plot_type == "计数图":
                sns.countplot(x=df[analysis_col], ax=ax, palette=sns.color_palette('Blues'))
                ax.set_ylabel("计数")
                ax.set_xlabel(analysis_col)
                plt.xticks(rotation=tick_rotation, ha="right", fontsize=tick_fontsize)
                plt.tight_layout()
            else:
                st.warning("该图表类型不适用于数值型变量")
                pass
                
            # 应用坐标轴设置
            ax.set_xscale(x_scale)
            ax.set_yscale(y_scale)
            plt.xticks(rotation=tick_rotation, ha="right", fontsize=tick_fontsize)
            if plot_type in ["散点图", "折线图", "箱线图"] and 'y_range' in locals():
                ax.set_ylim(y_range)
                
            st.pyplot(fig)
    else:
        # 类别型变量
        if plot_type == "饼图":
            fig, ax = plt.subplots()
            labels = df[analysis_col].value_counts().index
            df[analysis_col].value_counts().plot.pie(
                autopct='%1.1f%%',
                ax=ax,
                textprops={
                    'fontsize': tick_fontsize,
                    'rotation_mode': 'anchor',
                    'ha': 'center',
                    'va': 'center'
                }
            )
            ax.set_ylabel('')
            plt.tight_layout()
            st.pyplot(fig)
        elif plot_type == "箱线图":
            fig, ax = plt.subplots()
            sns.boxplot(y=df[analysis_col], ax=ax)
            ax.set_ylabel(analysis_col)
            st.pyplot(fig)
        elif plot_type == "计数图":
            fig, ax = plt.subplots()
            sns.countplot(x=df[analysis_col], ax=ax, palette=sns.color_palette('Blues'))
            plt.xticks(rotation=tick_rotation, ha="right", fontsize=tick_fontsize)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.warning("该图表类型不适用于类别型变量")
    
    # 多变量分析 - 重新设计
    st.subheader("多变量分析")
    
    with st.expander("分析设置", expanded=True):
        # 变量选择
        selected_cols = st.multiselect(
            "选择分析变量2个(热力图可以选择多个)",
            df.columns,
            key="mv_cols"
        )
        
        # 图表类型选择
        mv_plot_type = st.selectbox(
            "图表类型",
            ["散点图", "热力图", "关系图", "小提琴图", "箱线图"],
            key="mv_plot_type"
        )
        
        # 通用设置
        tick_rotation = st.slider("X轴标签旋转角度", 0, 90, 45)
        fig_size = st.slider("图表大小", 5, 15, 10)
        
        # 热力图特殊设置
        if mv_plot_type == "热力图":
            corr_method = st.selectbox(
                "相关系数计算方法",
                ["pearson", "spearman"],
                key="corr_method"
            )
    
    # 生成图表
    if st.button("生成图表", key="mv_plot_btn"):
        if len(selected_cols) < 2:
            st.warning("请选择至少2个变量")
        elif len(selected_cols) > 4:
            st.warning("最多选择4个变量")
        else:
            plt.figure(figsize=(fig_size, fig_size*0.8))
            
            try:
                if mv_plot_type == "散点图":
                    if len(selected_cols) == 2:
                        sns.scatterplot(
                            x=selected_cols[0],
                            y=selected_cols[1],
                            data=df
                        )
                    else:
                        sns.pairplot(df[selected_cols])
                        
                elif mv_plot_type == "热力图":
                    corr = df[selected_cols].corr(method=corr_method)
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
                    
                elif mv_plot_type == "关系图":
                    if len(selected_cols) == 2:
                        g = sns.relplot(
                            x=selected_cols[0],
                            y=selected_cols[1],
                            data=df,
                            kind="line"
                        )
                        # 设置统一的样式
                        for ax in g.axes.flat:
                            ax.tick_params(axis='x', rotation=tick_rotation)
                        st.pyplot(g.figure)
                    else:
                        st.warning("关系图需要选择2个变量")
                    
                elif mv_plot_type in ["小提琴图", "箱线图"]:
                    if len(selected_cols) >= 2:
                        x_col = selected_cols[0]
                        y_col = selected_cols[1]
                        if mv_plot_type == "小提琴图":
                            sns.violinplot(x=x_col, y=y_col, data=df)
                        else:
                            sns.boxplot(x=x_col, y=y_col, data=df)
                
                # 统一设置标签和样式
                plt.xticks(rotation=tick_rotation)
                plt.tight_layout()
                st.pyplot(plt.gcf())
                
            except Exception as e:
                st.error(f"图表生成错误: {str(e)}")
            finally:
                plt.close()

    # 模型预测部分
    elif menu == "模型预测":
        st.header("模型预测")
        
        # 确保特征与训练时一致
        required_features = ['collegeGPA', 'age', 'hours']  # 示例特征，需根据实际模型调整
        missing_features = [f for f in required_features if f not in df.columns]
        
        if missing_features:
            st.error(f"缺少必要特征: {', '.join(missing_features)}")
        else:
            # 确保特征顺序一致
            X = df[required_features]
            
            # 加载模型并进行预测
            try:
                model = joblib.load('model.pkl')
                predictions = model.predict(X)
                df['预测结果'] = predictions
                st.dataframe(df)
            except Exception as e:
                st.error(f"预测失败: {str(e)}")

# 数据建模模块
elif menu == "数据建模" and st.session_state.df is not None:
    st.header("数据建模")
    df = st.session_state.df
    
    # 第一步：选择建模类型
    st.subheader("选择建模类型")
    model_category = st.selectbox("请先选择建模类型",
        ["分类", "聚类", "回归"])
    
    # 第二步：选择特征和目标
    st.subheader("选择特征和目标")
    features = st.multiselect("选择特征列", df.columns)
    
    # 分类和回归需要目标列，聚类不需要
    if model_category in ["分类", "回归"]:
        target = st.selectbox("选择目标列", df.columns)
    else:
        target = None
    
    if features and (target is not None or model_category == "聚类"):
        # 数据预处理
        X = df[features]
        y = df[target] if target else None
        
        # 处理分类变量
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        if not cat_cols.empty:
            X = pd.get_dummies(X, columns=cat_cols)
        
        # 确保目标变量是数值型（分类问题）
        if model_category == "分类":
            y = y.astype('category').cat.codes
        
        # 划分训练测试集（聚类不需要y）
        test_size = st.slider("测试集比例", 0.1, 0.5, 0.2)
        if model_category == "聚类":
            X_train, X_test = train_test_split(
                X, test_size=test_size, random_state=42)
            y_train = y_test = None
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)
        
        # 模型训练
        st.subheader("模型选择")
        
        if model_category == "分类":
            model_type = st.selectbox("选择分类算法",
                ["线性判别分析", "逻辑回归", "决策树", "KNN", "随机森林", "支持向量机"])
        elif model_category == "聚类":
            model_type = st.selectbox("选择聚类算法",
                ["K-Means", "凝聚聚类", "DBSCAN"])
        else:  # 回归
            model_type = st.selectbox("选择回归算法",
                ["最小二乘法", "随机森林回归", "支持向量回归"])
        
        if st.button("训练模型"):
            model = None  # 初始化model变量
            try:
                if model_category == "分类":
                    if model_type == "线性判别分析":
                        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                        model = LinearDiscriminantAnalysis()
                    elif model_type == "逻辑回归":
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression()
                    elif model_type == "决策树":
                        from sklearn.tree import DecisionTreeClassifier
                        model = DecisionTreeClassifier()
                    elif model_type == "KNN":
                        model = KNeighborsClassifier()
                    elif model_type == "随机森林":
                        model = RandomForestClassifier()
                    elif model_type == "支持向量机":
                        model = SVC()
                elif model_category == "回归":
                    if not pd.api.types.is_numeric_dtype(y_train):
                        y_train = pd.to_numeric(y_train, errors='coerce')
                    if model_type == "随机森林回归":
                        model = RandomForestRegressor()
                    elif model_type == "支持向量回归":
                        model = SVR()
                    else:  # 最小二乘法
                        model = LinearRegression()
                else:  # 聚类
                    if model_type == "K-Means":
                        model = KMeans()
                    elif model_type == "凝聚聚类":
                        model = AgglomerativeClustering()
                    elif model_type == "DBSCAN":
                        model = DBSCAN()
                    else:  # 默认K-Means
                        model = KMeans()
                
                if model_category == "聚类":
                    y_pred = model.fit_predict(X_train)
                    # 对于聚类，我们使用fit_predict的结果作为预测
                    if hasattr(model, 'labels_'):
                        y_pred = model.labels_
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
            except Exception as e:
                st.error(f"模型训练失败: {str(e)}")
                st.stop()
            st.session_state.model = model
            st.success(f"{model_type}模型训练完成")
            
            # 模型导出
            model_bytes = io.BytesIO()
            joblib.dump(model, model_bytes)
            model_bytes.seek(0)
            
            st.download_button(
                label="下载模型",
                data=model_bytes,
                file_name=f"{model_type}_model.joblib",
                mime="application/octet-stream"
            )
        
        # 模型评估
        if st.session_state.model is not None:
            st.subheader("模型评估")
            if model_category == "聚类":
                if hasattr(st.session_state.model, 'fit_predict'):
                    y_pred = st.session_state.model.fit_predict(X_test)
                elif hasattr(st.session_state.model, 'labels_'):
                    y_pred = st.session_state.model.labels_
                else:
                    y_pred = st.session_state.model.predict(X_test)
            else:
                y_pred = st.session_state.model.predict(X_test)
            
            if model_category == "分类":
                from sklearn.metrics import accuracy_score
                acc = accuracy_score(y_test, y_pred)
                st.write(f"准确率: {acc:.2%}")
            elif model_category == "回归":
                st.text("回归评估:")
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"均方误差(MSE): {mse:.4f}")
                fig, ax = plt.subplots(figsize=(8,6))
                sns.regplot(x=y_test, y=y_pred, ax=ax)
                ax.set_title("回归预测 vs 实际值", pad=20)
                st.pyplot(fig, clear_figure=True)
                plt.close()
            else:
                # 模型评估
                st.subheader("模型评估")
                try:
                    if model_category == "分类":
                        # 确保目标变量是分类类型
                        if pd.api.types.is_numeric_dtype(y_test):
                            unique_values = len(y_test.unique())
                            if unique_values > 10:  # 数值型变量值过多，不适合分类
                                raise ValueError("分类目标变量包含过多唯一值，请确认是否为分类问题")
                            y_test = y_test.astype('category').cat.codes
                            y_pred = y_pred.astype('category').cat.codes
                        
                        report = classification_report(y_test, y_pred, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose())
                    
                    elif model_category == "回归":
                        if not pd.api.types.is_numeric_dtype(y_test):
                            y_test = pd.to_numeric(y_test, errors='coerce')
                        mse = mean_squared_error(y_test, y_pred)
                        st.write(f"均方误差(MSE): {mse:.4f}")
                        fig, ax = plt.subplots(figsize=(8,6))
                        sns.regplot(x=y_test, y=y_pred, ax=ax)
                        ax.set_title("回归预测 vs 实际值", pad=20)
                        st.pyplot(fig, clear_figure=True)
                        plt.close()
                    
                    else:  # 聚类
                        try:
                            silhouette = silhouette_score(X_test, y_pred)
                            st.write(f"轮廓系数: {silhouette:.4f}")
                        except ValueError as ve:
                            st.warning(f"聚类评估失败: {str(ve)}")
                
                except Exception as e:
                    st.error(f"模型评估出错: {str(e)}")
                    st.warning("请检查: 1) 目标变量类型是否正确 2) 数据是否包含有效值")
            

    
    # 数据预览模块
elif menu == "数据预览" and st.session_state.df is not None:
    st.header("数据预览")
    df = st.session_state.df
    
    # 创建数据预览和信息展示的标签页
    tab1, tab2, tab3, tab4 = st.tabs(["数据预览", "数据信息", "描述统计", "唯一值"])
    
    with tab1:
        st.dataframe(df.head())
        
    with tab2:
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
    with tab3:
        st.dataframe(df.describe())
        
    with tab4:
        col = st.selectbox("选择要查看唯一值的列", df.columns)
        if df[col].nunique() < 100:  # 避免显示过多唯一值
            st.write("唯一值列表:")
            st.write(df[col].unique())
            st.write("值计数:")
            st.write(df[col].value_counts())
        else:
            st.warning(f"该列有{df[col].nunique()}个唯一值，数量过多不便显示")

# 无数据提示
elif st.session_state.df is None and menu != "数据导入":
    st.warning("请先导入数据")