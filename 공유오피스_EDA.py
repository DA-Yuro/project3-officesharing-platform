!pip install koreanize_matplotlib -q

import koreanize_matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score, root_mean_squared_log_error

site_path = r'/content/drive/MyDrive/코드잇_데이터분석_6기/99_프로젝트/03_중급프로젝트(공유오피스체험결제)/DATA/site_area.csv'
access_path = r'/content/drive/MyDrive/코드잇_데이터분석_6기/99_프로젝트/03_중급프로젝트(공유오피스체험결제)/DATA/trial_access_log.csv'
payment_path = r'/content/drive/MyDrive/코드잇_데이터분석_6기/99_프로젝트/03_중급프로젝트(공유오피스체험결제)/DATA/trial_payment.csv'
regist_path = r'/content/drive/MyDrive/코드잇_데이터분석_6기/99_프로젝트/03_중급프로젝트(공유오피스체험결제)/DATA/trial_register.csv'
visit_path = r'/content/drive/MyDrive/코드잇_데이터분석_6기/99_프로젝트/03_중급프로젝트(공유오피스체험결제)/DATA/trial_visit_info.csv'

# 지점별 면적 테이블
area_df = pd.read_csv(site_path)
area_df.tail(15)

# 3일체험신청 테이블
register_df = pd.read_csv(regist_path)
register_df.tail()

register_df.info()

# 3일체험 신청자 일자별 방문기록
visit_df = pd.read_csv(visit_path)
visit_df.tail()

visit_df['site_id'].unique()

visit_df['user_uuid'].nunique()

visit_df.info()

# 3일체험 신청자 신청자 결제여부
pay_df = pd.read_csv(payment_path)
pay_df.tail()

pay_df['is_payment'].unique()

register_df.info()

# 전체 중복값 제거(중복신청오류로 판단)
register_df = register_df.drop_duplicates()

# 신청 날짜가 다르지만 한 유저의 중복신청한 인원 7명
register_df[register_df['user_uuid'].duplicated()]

# 방문신청한 사람들 총 9624명
register_unique_df = register_df[~register_df['user_uuid'].duplicated(keep='first')] # 날짜가 다르더라도 중복신청한 인원 제외
register_unique_df.info()

# 신청후 방문까지 한 유저 6534명
register_visit_df = register_unique_df[register_unique_df['user_uuid'].isin(visit_df['user_uuid'])]
register_visit_df.info()

# 3일체험 신청자 일자별 방문기록
visit_df.info()

# 중복확인
visit_df.duplicated().sum()

# 유저id 중복이 있지만 방문일자가 달라 중복제외는 하지 않음
visit_df[visit_df.duplicated()]

# 결측확인
visit_df.isnull().sum()

visit_null_df = visit_df[visit_df.isnull().any(axis=1)]
visit_null_df.info()

visit_null_pay = pd.merge(pay_df, visit_null_df, on='user_uuid', how='inner')
visit_null_pay.info()

# 출입기록이 없는 사람들의 결제이력을 봤을때, 생각보다 결제를 비율이 높음
visit_null_pay['is_payment'].value_counts()

visit_null_pay['user_uuid'].value_counts()

# 신청후 방문까지 한 유저 6534명 반영
# 확인결과 기존 방문테이블과 차이가 없음, 신청자는 모두 방문함
filter_visit_df = visit_df[visit_df['user_uuid'].isin(register_visit_df['user_uuid'])]
filter_visit_df.tail()

# 데이터 타임 변경
filter_visit_df['first_enter_time'] = filter_visit_df['first_enter_time'].str.slice(0, 19)
filter_visit_df['last_leave_time'] = filter_visit_df['last_leave_time'].str.slice(0, 19)

filter_visit_df['first_enter_time'] = pd.to_datetime(filter_visit_df['first_enter_time'])
filter_visit_df['last_leave_time'] = pd.to_datetime(filter_visit_df['last_leave_time'])

filter_visit_df.tail(2)

filter_visit_duplicate = filter_visit_df[filter_visit_df.duplicated(subset=None, keep='first')]
filter_visit_duplicate

filter_visit_cleaned = filter_visit_df.drop_duplicates(subset=None, keep='first')
filter_visit_cleaned.head(10)

# 방문 출입일시 중복확인
duplicated_enter = filter_visit_cleaned[filter_visit_cleaned.duplicated(subset=['user_uuid', 'first_enter_time'], keep=False)]
duplicated_enter.tail(10)

# 유저별 머문시간 총합계산
user_stay_sum = filter_visit_cleaned.groupby('user_uuid')['stay_time_second'].sum().reset_index()
user_stay_sum

# 박스플롯으로 이상치 확인
sns.boxplot(data=user_stay_sum, x='stay_time_second')
plt.title('머문시간(초)')
plt.show()

user_stay_sum.describe()

# 15분이하로 머문 유저 정의
user_under = user_stay_sum[user_stay_sum['stay_time_second'] <= 900]

# 내림차순으로 확인
# 너무 높은 수치로 e5e8feb2-5c4f-4b48-899d-f46d7a484d58 해당 유저만 제외
user_stay_sum.sort_values(by='stay_time_second', ascending=False).head(15)

# 15분 이하 유저 리스트
user_under_list = user_stay_sum[user_stay_sum['stay_time_second'] <= 900]['user_uuid'].tolist()

# 15분 이하 유저 + 너무 오래머문 이상치 e5e8feb2-5c4f-4b48-899d-f46d7a484d58
exclude_users = user_under_list + ['e5e8feb2-5c4f-4b48-899d-f46d7a484d58']

# 제외해서 필터링
filtered_visit = filter_visit_cleaned[~filter_visit_cleaned['user_uuid'].isin(exclude_users)]

filtered_visit.info()

# 박스플롯으로 이상치 확인
sns.boxplot(data=filtered_visit, x='stay_time_second')
plt.title('머문시간(초)')
plt.show()

# 결측치 제외
filtered_visit_drop = filtered_visit.copy()
filtered_visit_drop.dropna(inplace=True)
filtered_visit_drop.info()

# 1. user_uuid별 site_id 고유 개수 세기
site_count = filtered_visit_drop.groupby('user_uuid')['site_id'].nunique().reset_index()

# 2. 2개 이상 방문했으면 1, 아니면 0
site_count['multi_site_user'] = (site_count['site_id'] >= 2).astype(int)

# 3. 병합 전 혹시라도 기존에 multi_site_user 컬럼이 있으면 제거
filtered_visit_drop = filtered_visit_drop.drop(columns=['multi_site_user'], errors='ignore')

# 4. site_count에서 필요한 컬럼만 붙이기
filtered_visit_drop2 = filtered_visit_drop.merge(
    site_count[['user_uuid', 'multi_site_user']],
    on='user_uuid',
    how='left'
)

filtered_visit_drop2.head()

filtered_visit_drop2['multi_site_user'].value_counts()

filtered_visit_drop2['user_uuid'].nunique()

filtered_visit_area = filtered_visit_drop2.copy()
filtered_visit_area.tail(1)

area_df.head(15)

# site_id를 면적별로 맵핑
area = {
    1: 50,
    2: 100,
    3: 150,
    4: 100,
    5: 150,
    6: 150,
    17: 50,
    47: 50,
    49: 50
        }

# 칼럼명 변경
filtered_visit_area.rename(columns={'site_id':'site_area'}, inplace=True)

# 맵핑 데이터 적용
filtered_visit_area['site_area'] = filtered_visit_area['site_area'].replace(area)
filtered_visit_area['site_area'].unique()

filtered_visit_area.tail(2)

filtered_visit_area['user_uuid'].nunique()

filtered_visit_area[filtered_visit_area['user_uuid'] == 'acf3e288-4487-492b-9477-df149fb72e83']

area_counts = filtered_visit_area.groupby('user_uuid')['site_area'].nunique()

# 유저별 여러지점 방문 확인
multi_area_users = area_counts[area_counts > 1].index
result = filtered_visit_area[filtered_visit_area['user_uuid'].isin(multi_area_users)]

result

# site_area 칼럼 제거
filtered_visit_area.drop(columns='site_area', inplace=True)

visitend_df = filtered_visit_area.copy()

# stay_time_second를 합산했을 때, 묶이지 않는 칼럼제거
visitend_df.drop(columns=['date', 'stay_time', 'last_leave_time'], inplace=True)

visit_summary = visitend_df.groupby('user_uuid').agg({
    'stay_time_second': 'sum',
    'first_enter_time': 'min',
    'multi_site_user': 'max'
}).reset_index()

visit_summary.head()

visit_summary['user_uuid'].nunique()

pay_df.info()

pay_duple = pay_df[pay_df.duplicated(subset=None, keep=False)]
pay_duple

pay_cleaned = pay_df.drop_duplicates(subset=None, keep='first')
pay_cleaned.info()

visit_summary_pay = pd.merge(visit_summary, pay_cleaned, on='user_uuid', how='inner')
visit_summary_pay.info()

visit_summary_pay.head()

(visit_summary_pay['is_payment'] == 0).sum()

(visit_summary_pay['is_payment'] == 1).sum()

visit_summary_pay['user_uuid'].nunique()

visit_summary_pay['hour'] = visit_summary_pay['first_enter_time'].dt.hour

visit_summary_pay.head()

visit_summary_pay2 = visit_summary_pay.copy()

visit_summary_pay2.drop(columns=['user_uuid', 'first_enter_time'], inplace=True)

corr_with_pay = visit_summary_pay2.corr()['is_payment'].sort_values(ascending=False)
corr_with_pay

plt.figure(figsize=(8, 5))
sns.heatmap(visit_summary_pay2.corr(), annot=True, fmt='.2f', cmap='coolwarm')

plt.title('결제의 상관관계 히트맵')
plt.show()

X = visit_summary_pay2.drop(columns='is_payment')
y = visit_summary_pay2['is_payment']

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# 모델학습
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RMSE: {rmse:.3f}')
print(f'R2: {r2:.3f}')

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

X = visit_summary_pay2.drop(columns='is_payment')
y = visit_summary_pay2['is_payment']

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# 모델 학습
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# 모델 평가
y_pred = tree.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'acc score: {acc:.3f}')

# 트리 결과 시각화
plt.figure(figsize=(12, 8))
plot_tree(tree,
          feature_names=X_train.columns.tolist(),
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True,
          fontsize=10
          )
plt.title('decision tree')
plt.show()

# 트리가 너무 복잡 -> 하이퍼파라미터 실험
# 실험할 하이퍼파라미터 값
depths = [2, 3, 5, 7, 10, 20, 30]
splits = [2, 10, 20, 30, 60, 80, 200]
leaves = [1, 2, 3, 5, 10, 15]

# 깊이에 따른 모델 정확도 비교
for d in depths:
    clf = DecisionTreeClassifier(max_depth=d, random_state=25)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'accuracy: {acc:.3f}')

# 분할 샘플에 따른 모델 정확도 비교
for s in splits:
    clf = DecisionTreeClassifier(min_samples_split=s, random_state=24)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'accuracy: {acc:.3f}')

# 리프 수 기준에 따른 모델 정확도 비교
for leaf in leaves:
    clf = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=24)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'accuracy: {acc:.3f}')

# 하이퍼파라미터 적용 모델 학습
tree = DecisionTreeClassifier(
    max_depth = 3,
    min_samples_split = 200,
    min_samples_leaf = 5,
)
tree.fit(X_train, y_train)

# 모델 평가
y_pred = tree.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'acc score: {acc:.3f}')

# 트리 결과 시각화
plt.figure(figsize=(12, 8))
plot_tree(tree,
          feature_names=X_train.columns.tolist(),
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True,
          fontsize=10
          )
plt.title('decision tree')
plt.show()

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    oob_score=True
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
acc_rf = accuracy_score(y_test, rf_pred)

print(f'랜덤포레스트 모델 정확도:{acc_rf:.3f}')

# 설정할 하이퍼파라미터 값을 딕셔너리 형태로 선언
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [None, 10],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 3]
}

# 모델준비
rf = RandomForestClassifier(n_jobs=-1, random_state=25)

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(
    rf,
    param_grid=param_grid, # 리스트값을 가지는 딕셔너리 형태
    cv=5, # cross validation 교차검증
    scoring='accuracy', # 예측 성능 지표 설정
    n_jobs=-1, # cpu 성능최대
    refit=True, # 최적의 하이퍼파라미터를 찾은 뒤 해당값으로 재학습
    verbose=1 # 학습 과정 출력 정도(0, 1, 2) -> 0은 출력안함 1은 어느정도 출력 2는 자세히 출력
)

gs.fit(X_train, y_train)

best_rf = gs.best_estimator_
best_pred = best_rf.predict(X_test)

best_acc = accuracy_score(y_test, best_pred)
print(f'베스트 정확도:{best_acc:.3f}')

feat_names = X_train.columns.tolist()

# 피처 중요도를 시리즈 데이터 타입으로 저장
# corr 즉 상관관계랑은 좀 다르다
# 모델학습에 중요한 역할을 한 것들
importances = pd.Series(best_rf.feature_importances_, index=feat_names).sort_values(ascending=False)

# 상위 10개 시각화
top_n = 10
top_importances = importances.head(top_n)

# 역순으로 그려서 가장 큰 값이 위로 가도록 설정
top_importances.iloc[::-1].plot(kind='barh')

top_importances

b_rf = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    oob_score=True,
    max_depth=10,
    min_samples_leaf=3,
    random_state=25
)

b_rf.fit(X_train, y_train)
b_rf_pred = b_rf.predict(X_test)
acc_b_rf = accuracy_score(y_test, b_rf_pred)

print(f'베스트 정확도:{acc_b_rf:.3f}')

from sklearn.metrics import classification_report

# 테스트 데이터에 대해 예측 및 성능 평가
b_rf_pred = b_rf.predict(X_test)
print(classification_report(y_test, b_rf_pred))

b_rf2 = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    oob_score=True,
    max_depth=10,
    min_samples_leaf=3,
    random_state=25,
    class_weight='balanced'
)

b_rf2.fit(X_train, y_train)
b_rf_pred2 = b_rf.predict(X_test)
acc_b_rf2 = accuracy_score(y_test, b_rf_pred2)

print(f'베스트 정확도:{acc_b_rf2:.3f}')

# 테스트 데이터에 대해 예측 및 성능 평가
b_rf_pred2 = b_rf2.predict(X_test)
print(classification_report(y_test, b_rf_pred2))

# 트리 결과 시각화

estimator = b_rf.estimators_[0]

plt.figure(figsize=(50, 15))
plot_tree(estimator,
          feature_names=X_train.columns.tolist(),
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=5
          )
plt.title('RandomForestClassifier0')
plt.show()

# 트리 결과 시각화

estimator2 = b_rf2.estimators_[0]

plt.figure(figsize=(50, 15))
plot_tree(estimator2,
          feature_names=X_train.columns.tolist(),
          class_names=['No', 'Yes'],
          filled=True,
          rounded=True,
          fontsize=10,
          max_depth=7
          )
plt.title('RandomForestClassifier0')
plt.show()

from sklearn.metrics import f1_score

# 최적 모델을 이용해 예측
y_pred = b_rf.predict(X_test)

# F1 스코어 계산 (이진 분류 기준)
f1 = f1_score(y_test, y_pred)

print(f"F1 Score: {f1:.3f}")

from sklearn.metrics import (
    accuracy_score, confusion_matrix
)

cm = confusion_matrix(y_test, y_pred)
cm

acc = accuracy_score(y_test, y_pred)
print(f'acc: {acc}')

from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.show()

from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report

# XGBoost 모델 생성
xgb_model = XGBClassifier(
    max_depth=10,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=25,
    scale_pos_weight=4,
    use_label_encoder=False,
    eval_metric='logloss'  # 경고 방지용
)

# 학습
xgb_model.fit(X_train, y_train)

# 예측
y_pred = xgb_model.predict(X_test)

# F1 스코어
f1 = f1_score(y_test, y_pred)
print(f"XGBoost F1 Score: {f1:.4f}")

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# 1. 데이터 분리
df = visit_summary_pay2.copy()
paid = df[df['is_payment'] == 1].drop(columns=['is_payment'])
unpaid = df[df['is_payment'] == 0].drop(columns=['is_payment'])

# 2. 피처 스케일링
scaler = StandardScaler()
X_paid = scaler.fit_transform(paid)
X_unpaid = scaler.fit_transform(unpaid)

# 3. 최적 클러스터 개수(k) 찾기 — 엘보우 방법
def find_k(X, name):
    sse = []
    K = range(2, 9)
    for k in K:
        sse.append(KMeans(n_clusters=k, random_state=42).fit(X).inertia_)
    plt.plot(K, sse, marker='o', label=name)

plt.figure(figsize=(8, 5))
find_k(X_paid, 'Paid')
find_k(X_unpaid, 'Unpaid')
plt.xlabel('Number of clusters k')
plt.ylabel('Sum of squared distances (inertia)')
plt.legend()
plt.title('Elbow Method')
plt.show()

# 4. 클러스터링 및 PCA 시각화
def cluster_and_plot(X, name, k):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(6, 5))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.title(f'{name} group clustering (k={k})')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.show()
    return km, labels

# (예시) k=3으로 설정
km_paid, labels_paid = cluster_and_plot(X_paid, 'Paid', k=3)
km_unpaid, labels_unpaid = cluster_and_plot(X_unpaid, 'Unpaid', k=3)

# 5. 결과를 원래 데이터프레임에 병합
df.loc[df['is_payment'] == 1, 'cluster'] = labels_paid
df.loc[df['is_payment'] == 0, 'cluster'] = labels_unpaid

# Paid 그룹에 클러스터 정보 병합
paid['cluster'] = labels_paid

# PCA 적용
pca = PCA(n_components=2, random_state=42)
X_pca_paid = pca.fit_transform(X_paid)

# 시각화
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca_paid[:, 0], y=X_pca_paid[:, 1],
                hue=paid['cluster'].astype(int),
                palette='Set2', s=70)

plt.title('결제 유저 클러스터링 (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 클러스터별 평균값 요약
paid_cluster_summary = paid.groupby('cluster').mean(numeric_only=True).round(2)

# 보기 좋게 출력
display(paid_cluster_summary)

paid['cluster'].value_counts()

paid2 = paid.copy()

paid2.drop(columns='cluster', inplace=True)

# 군집 특성을 파악하기 위해 로딩확인
# PCA
pca = PCA(n_components=3)
pca.fit(paid2)

# PC1의 로딩 확인
loadings = pca.components_[0]  # PC1 벡터
for var, loading in zip(paid2.columns, loadings):
    print(f"{var}: {loading}")

# unpaid 그룹에 클러스터 정보 병합
unpaid['cluster'] = labels_unpaid

# PCA 적용
pca = PCA(n_components=2, random_state=42)
X_pca_unpaid = pca.fit_transform(X_unpaid)

# 시각화
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca_unpaid[:, 0], y=X_pca_unpaid[:, 1],
                hue=unpaid['cluster'].astype(int),
                palette='Set2', s=70)

plt.title('비결제 유저 클러스터링 (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 클러스터별 평균값 요약
unpaid_cluster_summary = unpaid.groupby('cluster').mean(numeric_only=True).round(2)

# 보기 좋게 출력
display(unpaid_cluster_summary)

unpaid['cluster'].value_counts()
