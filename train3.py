import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor

# 忽略警告，保持输出整洁
warnings.filterwarnings('ignore')

# --- 1. 读取数据 ---
train_path = r"D:/AllData/competitions/playground-series-s6e1/playground-series-s6e1/train.csv"
test_path = r"D:/AllData/competitions/playground-series-s6e1/playground-series-s6e1/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 保存 ID 用于最后提交
test_ids = test_df['id']

# --- 2. 核心步骤：特征工程 (Feature Engineering) ---
# 这是提分的关键！我们需要“教”模型理解数据
def engineer_features(df):
    df = df.copy()
    
    # 2.1 处理序数特征 (Ordinal Features)
    # 睡眠质量是有顺序的，不要用 OneHot，直接转数字
    sleep_map = {'poor': 0, 'average': 1, 'good': 2}
    # 加上 fillna 防止测试集有空值报错
    df['sleep_quality_num'] = df['sleep_quality'].map(sleep_map).fillna(1)
    
    # 2.2 创造“交互特征”
    # 学习/睡眠比：衡量你是高效学习还是疲劳战
    # 加 0.1 是为了防止除以 0
    df['study_sleep_ratio'] = df['study_hours'] / (df['sleep_hours'] + 0.1)
    
    # 总投入度：学习时间 * 出勤率
    df['total_dedication'] = df['study_hours'] * (df['class_attendance'] / 100)
    
    # 设施与分数的潜在关系 (假设设施评分也是有序的 low/medium/high)
    facility_map = {'low': 0, 'medium': 1, 'high': 2}
    df['facility_score'] = df['facility_rating'].map(facility_map).fillna(1)
    
    # 2.3 清理掉已经被转化过的原始列，防止多重共线性
    df = df.drop(columns=['sleep_quality', 'facility_rating', 'id'], errors='ignore')
    
    return df

# 对训练集和测试集同时应用特征工程
print("正在进行特征工程...")
X = engineer_features(train_df.drop(columns=['exam_score']))
y = train_df['exam_score']
X_test = engineer_features(test_df)

# --- 3. 重新定义预处理管道 ---
# 此时我们的列已经变了，需要重新区分数值和类别
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"数值特征: {len(numeric_features)} 个")
print(f"类别特征: {len(categorical_features)} 个")

# 数值处理：填补缺失值 + 标准化
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 类别处理：填补缺失值 + 独热编码
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# --- 4. 定义模型融合 (Ensemble) ---
# 混合 XGBoost (强力) 和 Random Forest (稳定)

# 4.1 XGBoost 模型 (GPU加速)
xgb_params = {
    'n_estimators': 2000,
    'learning_rate': 0.01,    # 降低学习率，增加树的数量，通常能提高精度
    'max_depth': 6,
    'subsample': 0.8,         # 每次只用80%的数据，防止过拟合
    'colsample_bytree': 0.8,  # 每次只用80%的特征
    'tree_method': 'hist',
    'device': 'cuda',         # GPU 加速
    'random_state': 42,
    'n_jobs': -1
}

xgb_model = XGBRegressor(**xgb_params)

# 4.2 随机森林模型 (作为补充)
# 注意：随机森林没有 device='cuda' 参数，除非用 cuml，这里用 CPU 并行即可
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=4,       # 叶子节点最少样本数，防止过拟合
    n_jobs=-1,
    random_state=42
)

# 4.3 投票回归器 (Voting)
# weights=[2, 1] 表示我们更信任 XGBoost，给它 2 倍权重
ensemble_model = VotingRegressor(
    estimators=[('xgb', xgb_model), ('rf', rf_model)],
    weights=[3, 1] 
)

# --- 5. 训练与验证 ---
# 构建最终管道
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('model', ensemble_model)])

print("开始训练集成模型 (XGBoost + Random Forest)...")

# 我们可以先做一个简单的交叉验证看看分数提升没
# 这里的 negative_root_mean_squared_error 是负数，取绝对值看
cv_scores = cross_val_score(final_pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error')
print(f"本地交叉验证 RMSE 均值: {-cv_scores.mean():.4f}")

# 正式训练全量数据
final_pipeline.fit(X, y)
print("全量数据训练完成！")

# --- 6. 预测与生成文件 ---
print("正在生成预测结果...")
final_predictions = final_pipeline.predict(X_test)

# 防止预测出负数或超过100分（如果这是百分制考试）
final_predictions = np.clip(final_predictions, 0, 100)

submission = pd.DataFrame({
    'id': test_ids,
    'exam_score': final_predictions
})

save_path = "D:/AllData/competitions/playground-series-s6e1/playground-series-s6e1/submission_ensemble_v2.csv"
submission.to_csv(save_path, index=False)
print(f"恭喜！优化后的预测文件已生成: {save_path}")