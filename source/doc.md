# Chemical Process AI Dashboard – 팀 프로젝트 역할 분담 설계

## 팀 인원 : 3명

| 역할                    | 담당자 | 핵심 책임                           |
| --------------------- | --- | ------------------------------- |
| **① Data Engineer**   | 본인 ★  | 데이터 구조 설계 · 전처리 · 품질 관리 · 공정 해석 |
| **② Process Analyst** | B   | 알람 로직 설계 · 불량 예측 모델 개발          |
| **③ Frontend / PM**   | C   | Streamlit UI · 전체 구조 관리         |

---

## ① Data Engineer (데이터 엔지니어) - ★

**주요 업무**
- OK / NG 데이터 구조 정리
- 컬럼 정합성, 결측치, 이상치 점검
- 데이터 분포 분석 리포트 작성
- 예측 임계 확률 기준 설정

**목표**
- `data_preprocessing.ipynb`
- 컬럼 설명서
- Temp–Viscosity 상관관계 보고서
- Risk Level 기준 정의

---

## ② Process Analyst (공정 분석 담당)

**주요 업무**
- μ ± kσ 알람 로직 설계
- k 값 민감도 분석
- 시나리오 정의
- Logistic Regression 모델 구현
- 성능 평가 (Confusion Matrix)


**목표**
- 알람 로직 문서
- k 변화별 불량 탐지 정확도 표
- 정확도 / 재현율 보고서
- `ng_prediction_model.py`   

---

## ③ Frontend / PM (Streamlit & 총괄)

**주요 업무**
- Streamlit UI 구조 설계
- Sidebar Mode 분기
- 전체 코드 통합
- GitHub 관리

**목표**
- `app.py`
- UI 흐름 다이어그램
- 배포 링크 관리


---
---


# Chemical Process Monitoring & NG Prediction Dashboard

### using Python · Streamlit · Logistic Regression

---

## 1. 프로젝트 개요

본 프로젝트는 화학 공정 데이터(온도, 점도)를 기반으로

- 정상 / 불량 분포 시각화
- 공정 이상 알람 시스템
- 불량 발생 확률(Logistic Regression) 예측 모델

을 하나의 **통합 웹 대시보드**로 구현하는 것을 목표로 한다.

---

## 2. 사용 데이터

|컬럼명|설명|
|---|---|
|Lot|생산 로트 번호|
|Temp(°C)|공정 온도|
|Viscosity(cP)|점도|
|Failure|0=정상, 1=불량|

```python
ok = pd.read_csv("data/Chemical_Numeric_Data_OK.csv", encoding="cp949")
ng = pd.read_csv("data/Chemical_Numeric_Data_NG.csv", encoding="cp949")
df = pd.concat([ok, ng], ignore_index=True)
```

---

## 3. UI 구조 설계

### ▶ Sidebar : Mode Selector

```python
mode = st.sidebar.radio(
    "Select Dashboard Mode",
    [
        "📊 Data Status",
        "🚨 Alarm Threshold Setting",
        "🤖 NG Probability Prediction"
    ]
)
```

**단일 선택 구조**로 설계하여  
한 번에 하나의 기능만 명확하게 사용 가능하도록 했다.

---

## 4. Data Status 모드

### 목적

정상 / 불량 데이터 분포 구조를 직관적으로 파악.

### OK / NG 테이블 미리보기

```python
cols = ["Temp(°C)", "Viscosity(cP)"]

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🟢 OK Sample Data")
    st.dataframe(df[df["Failure"]==0][cols].head(10))

with col2:
    st.markdown("### 🔴 NG Sample Data")
    st.dataframe(df[df["Failure"]==1][cols].head(10))
```

### KDE 분포 시각화

```python
c1, c2 = st.columns(2)

with c1:
    sns.kdeplot(df[df["Failure"]==0]["Temp(°C)"], fill=True)
    sns.kdeplot(df[df["Failure"]==1]["Temp(°C)"], fill=True)

with c2:
    sns.kdeplot(df[df["Failure"]==0]["Viscosity(cP)"], fill=True)
    sns.kdeplot(df[df["Failure"]==1]["Viscosity(cP)"], fill=True)
```

---

## 5. Alarm Threshold Setting 모드

### 핵심 개념

정상 데이터 기준

> $μ±kσ\mu \pm k\sigmaμ±kσ$

범위를 벗어나는 순간 알람 발생.

```python
k = st.sidebar.slider("Sigma multiplier (k)", 1.0, 4.0, 3.0, 0.1)
feature = st.sidebar.selectbox("Select variable", ["Temp(°C)", "Viscosity(cP)"])
```

```python
mu = ok[feature].mean()
sigma = ok[feature].std()
lower = mu - k*sigma
upper = mu + k*sigma
```

```python
if latest < lower or latest > upper:
    st.error("🚨 ALARM")
else:
    st.success("✅ NORMAL")
```

---

## 6. NG Probability Prediction (AI 모델)

### 목적

> 공정 조건 입력 → **불량 발생 확률 사전 예측**

### 모델 학습

```python
X = df[["Temp(°C)", "Viscosity(cP)"]]
y = df["Failure"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)
```

### 예측 UI

```python
temp = st.slider("Temperature (°C)", float(X["Temp(°C)"].min()), float(X["Temp(°C)"].max()))
visc = st.slider("Viscosity (cP)", float(X["Viscosity(cP)"].min()), float(X["Viscosity(cP)"].max()))

input_data = scaler.transform([[temp, visc]])
prob = model.predict_proba(input_data)[0][1]
```

```python
st.metric("NG Probability", f"{prob*100:.2f} %")
```

---

## 7. 시스템 구조 요약

| 기능            | 역할          |
| ------------- | ----------- |
| Data Status   | 데이터 구조 이해   |
| Alarm Mode    | 사후 공정 이상 감지 |
| AI Prediction | 사전 불량 예방    |
