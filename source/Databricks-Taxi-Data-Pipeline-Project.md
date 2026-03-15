
# 0️⃣. 프로젝트 개요

>AWS S3에 저장된 Taxi 데이터를 Databricks에서 수집하고
>Medallion Architecture를 구축하여
>데이터 분석 및 예측 모델을 만드는 데이터 파이프라인 구현

## 0.1 사용 기술

- AWS S3
- Databricks
- Spark SQL
- Python
- Delta Lake

## 0.2 AWS S3를 선정이유

> S3는 데이터 레이크를 만들기 위한 가장 표준적인 저장소 이기 때문에 선정했습니다.

택시 요금 분석 프로젝트에서 데이터 레이크 아키텍처를 기반으로 설계했습니다.
스토리지딴(저장 계층)에는 AWS S3를 사용하여 원천 데이터를 저장했고, Databricks에서 메타데이터를 관리하게 했습니다.
이후 처리딴(처리 계층)으로 Databricks의 Spark를 활용하여 SQL과 Spark 기반 데이터 전처리를 수행했습니다.
이를 통해 Raw 데이터를 정제하여 분석 가능한 데이터셋으로 변환하는 구조로 아키텍처를 설계했습니다.

## 0.3 .parquet을 선정이유

> parquet은 컬럼 기반 저장 방식이라 분석 쿼리에서 필요한 컬럼만 읽을 수 있어 대표적인 csv보다 매우 빠르기 때문에 선정했습니다.

- parquet의 특징

| 특징         | 설명              |
| ---------- | --------------- |
| 컬럼 기반 저장   | 컬럼 단위로 저장       |
| 압축 효율 높음   | 동일 데이터 반복 압축    |
| 필요한 컬럼만 읽기 | I/O 감소          |
| 분석 최적화     | Spark SQL 성능 좋음 |

csv는 Row 기반 저장이기 때문에 전체 행을 읽어야 합니다.
반면 parquet은 Column 기반 저장이기 때문에 분석하는 입장에서 Read 수행시 I/O가 대폭 감소합니다.
또한 포맷 중에서도 압축이 매우 잘 된다고 알고있어. 저 자본, 고 효율이기 때문입니다. 

또 Spark 사용시 parquet을 읽을 때 추가 최적화도 하기 때문입니다.
parquet은 file metadata를 확인하고 필요 없는 데이터는 읽지 않습니다. (= Predicate Pushdown)

결론적으로 전체적인 cost적으로 다른 포맷들 중 상위권을 차지하기에 사용했습니다.

-> 추후 Delta Lake 프로젝트 진행 예정 (parquet의 상위 포맷)
`Delta Lake = Parquet + transaction log`


## 0.4 IAM 정책 json 필요 이유

> Databricks가 AWS S3 데이터에 접근할 수 있도록 IAM Role 기반 권한 구조를 구성했습니다.
> IAM 정책을 통해 S3 객체 조회 및 저장 권한을 부여하고 STS AssumeRole을 사용하여 Databricks가 해당 Role을 통해 안전하게 S3 데이터에 접근하도록 설계했습니다.

AWS는 기본적으로 모든 접근을 차단하는것을 시작으로 합니다. (`DENY`)
AWS를 수 많은 사람들이 사용하는 이유 중 하나로 저는 AWS 의 보안성이라고 생각합니다.
그렇기 때문에 Databricks에서 S3 버킷에 접근하려고 하면 초기에 `권한 없음` 으로 접근이 불가합니다.
해서 명시적 권한을 열어주기 위해 정책 편집기로 추가합니다.

```
-- Databricks Structure

Databricks Platform
        │
        │ 접근 요청
        ▼
AWS Account
        │
        ▼
S3 Bucket
```

IAM Role의 역할 = 임시 출입증
핵심 API = `sts:AssumeRole` = 역할 대신 수행

### 0.4.1 S3 접근 권한

```json
s3:GetObject     // 파일 읽기
s3:PutObject     // 파일 저장
s3:DeleteObject  // 파일 삭제
s3:ListBucket    // 버킷 목록 조휘
```

### 0.4.2 Role 위임 권한

```json
sts:AssumeRole   // Databricks가 IAM Role을 사용 가능
```


### 0.4.3 sns / sqs

Databricks가 S3에 파일을 가지고 있는지 물음(Polling)
데이터를 주기적으로 읽어 들일때마다 Polling 발생 (비용 증가, 새 파일 감지 속도 저하 = 비효율)

- `sns` = `Amazon Simple Notification Service` 이벤트 알림 브로드캐스트 
- `sqs` = `Amazon Simple Queue Service` 이벤트 메시지 저장
```
message:
{
 "file": "data_2026.parquet"
}
```

```
( S3 -------------> SNS -------------> SQS -------------> Databricks ) = Event-driven Architecture (Polling 문제 해결)
     파일 생성 이벤트
```

- Databricks 에서의 동작 변화
```
( SQS queue 확인 ---> 새 파일 이벤트 발견 ---> S3 파일 읽기 ---> Spark 처리 ) = 필요할 때만 읽는다. (속도 UP, 비용 DOWN)
```

- `sns` 정책 구성 (이벤트 알림 생성)
```json
sns:CreateTopic
sns:Publish
sns:Subscribe
```

- `sqs` 정책 구성 (이벤트 메시지 읽기)
```json
sqs:ReceiveMessage
sqs:SendMessage
sqs:GetQueueAttributes
```

- 변경된 아키텍처
```
Databricks Platform ---> AWS Account ---> S3 Bucket
                  접근 요청
데이터 업로드 ---> S3 ---> SNS ---> SQS ---> Databricks ---> Spark ---> Table
                    event
```


### 0.4.4 policy1 json 구조

>`policy1`은 Databricks가 IAM Role을 통해 S3 버킷의 데이터를 읽고, 쓰고, 관리할 수 있도록 최소한의 권한을 정의한 정책

- json
```
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Action": [
				"s3:GetObject",
				"s3:PutObject",
				"s3:DeleteObject",
				"s3:ListBucket",
				"s3:GetBucketLocation",
				"s3:ListBucketMultipartUploads",
				"s3:ListMultipartUploadParts",
				"s3:AbortMultipartUpload"
			],
			"Resource": [
				"arn:aws:s3:::<bucket name>/*",
				"arn:aws:s3:::<bucket name>"
			],
			"Effect": "Allow"
		},
		{
			"Action": [
				"sts:AssumeRole"
			],
			"Resource": [
				"arn:aws:iam::<account ID>:role/<bucket role>"
			],
			"Effect": "Allow"
		}
	]
}
```

- 구조
```
"Effect": "Allow"               // 허용한다. 3
"Action": [허용할 행동]          // []이 행동을 2
"Resource": [어디에 적용할지]     // [적용할 곳]에 1
```

1. `Statement 1`
- `Action`
```
{
 "Action": [
  "s3:GetObject", // 파일 읽기 ★
  "s3:PutObject", // 파일 저장 ★
  "s3:DeleteObject", // 파일 삭제 
  "s3:ListBucket", // 버킷 파일 목록 조회 ★
  "s3:GetBucketLocation", // 버킷 리전 확인
  
  // 대용량 파일 업로드
  "s3:ListBucketMultipartUploads",
  "s3:ListMultipartUploadParts",
  "s3:AbortMultipartUpload"
 ]
}
```

대용량 파일 업로드 시 필요한 권한
```
  "s3:ListBucketMultipartUploads",
  "s3:ListMultipartUploadParts",
  "s3:AbortMultipartUpload"
```
약 10GB parquet 파일을 part로 나누어 (part1, part2, ... ) 업로드 (= `Multipart Upload`)

- `Resource`
```
"Resource": [
 "arn:aws:s3:::<버킷이름>/*", // <버킷> 내부의 파일
 "arn:aws:s3:::<버킷이름>"    // <버킷> 자체 파일
]
```

2. `Statement 2` - 보안 구조
- `Action`
```
{
 "Action": ["sts:AssumeRole"]    // 이 Role을 다른 서비스(ex. Databricks)가 사용할 수 있음
}
```

- `Resource`
```
"Resource": [
	"arn:aws:iam::<account ID>:role/<bucket role>"  // <이 아이디의> <버킷 role> 에 적용한다.
]
```

- `Effect`
```
"Effect": "Allow" // 허락함
```


### 0.4.5 policy2 json 구조

> `policy2`는 Databricks가 S3 파일 이벤트를 감지하기 위해 SNS와 SQS 인프라를 생성하고 관리할 수 있도록 권한을 부여하는 정책

- json
```
{
	"Version": "2012-10-17",
	"Statement": [
		{
			"Sid": "ManagedFileEventsSetupStatement",
			"Effect": "Allow",
			"Action": [
				"s3:GetBucketNotification",
				"s3:PutBucketNotification",
				"sns:ListSubscriptionsByTopic",
				"sns:GetTopicAttributes",
				"sns:SetTopicAttributes",
				"sns:CreateTopic",
				"sns:TagResource",
				"sns:Publish",
				"sns:Subscribe",
				"sqs:CreateQueue",
				"sqs:DeleteMessage",
				"sqs:ReceiveMessage",
				"sqs:SendMessage",
				"sqs:GetQueueUrl",
				"sqs:GetQueueAttributes",
				"sqs:SetQueueAttributes",
				"sqs:TagQueue",
				"sqs:ChangeMessageVisibility",
				"sqs:PurgeQueue"
			],
			"Resource": [
				"arn:aws:s3:::<bucket name>",
				"arn:aws:sqs:*:*:csms-*",
				"arn:aws:sns:*:*:csms-*"
			]
		},
		{
			"Sid": "ManagedFileEventsListStatement",
			"Effect": "Allow",
			"Action": [
				"sqs:ListQueues",
				"sqs:ListQueueTags",
				"sns:ListTopics"
			],
			"Resource": [
				"arn:aws:sqs:*:*:csms-*",
				"arn:aws:sns:*:*:csms-*"
			]
		},
		{
			"Sid": "ManagedFileEventsTeardownStatement",
			"Effect": "Allow",
			"Action": [
				"sns:Unsubscribe",
				"sns:DeleteTopic",
				"sqs:DeleteQueue"
			],
			"Resource": [
				"arn:aws:sqs:*:*:csms-*",
				"arn:aws:sns:*:*:csms-*"
			]
		}
	]
}
```

- 구조
```
"Statement":[
	{생성},
	{조회},
	{삭제}
] // lifecycle 권한
```

- `"Sid": "ManagedFileEventsSetupStatement"` : 이벤트 파이프라인 생성 권한 중 가장 중요한 요소소
```
// S3 -> sns 이벤트 연결 설정
s3:GetBucketNotification 
s3:PutBucketNotification

// sns 이벤트 채널 생성
sns:CreateTopic
sns:Publish
sns:Subscribe

// sqs 이벤트 메시지 큐 생성
sqs:CreateQueue
sqs:SendMessage
sqs:ReceiveMessage
```

- `"Sid": "ManagedFileEventsListStatement"` : 리소스 조회 권한
```
// 이미 생성된 큐나 토픽 확인 -> Databricks가 이미 존재하는지 확인하기 위해 필요
sqs:ListQueues
sns:ListTopics
```

- `"Sid": "ManagedFileEventsTeardownStatement"` : 정리(clean up) 권한
```
// 임시 이벤트 리소스 삭제
sns:DeleteTopic
sqs:DeleteQueue
```

- `csms-*` : `csms = Cloud Storage Message Service`
```
// Databricks가 자동 생성하는 **리소스 이름 패턴**
// Databricks가 만든 큐만 관리하도록 제한하는 기능
arn:aws:sqs:*:*:csms-*
arn:aws:sns:*:*:csms-*
```


### 0.4.6 policy 1, 2 차이

`policy 1` 은 데이터 작업, 즉 S3 데이터에 접근하기 위한 정책 이고
`policy 2` 은 이벤트 감지 시스템으로 이벤트의 파이프라인을 관리한다.

## 0.5 Databricks 환경의 Medallion Architecture

> Bronze -> Silver -> Gold 구조
> 데이터 품질 관리와 재사용성을 위해 데이터를 단계적으로 정제하기 위한 데이터 레이크 설계 패턴

현실의 원천 데이터는 Null, 중복, 잘못된 값, 스키마 불일치와 같은 문제가 발생합니다.
이런 데이터를 가지고 분석을 하면 분석에 오류가 나거나 잘못된 결과를 가져옵니다.
그렇기 떄문에 단계적인 정제를 위해 Databricks 에서는 Medallion Architecture를 사용합니다.

나누는 이유
1. 데이터 품질 관리
	1. Raw 데이터 보호
	2. 문제 발생시 Bronze table로 돌아가 다시 처리
2. 재처리 가능
	1. Silver 로직 수정 시 Bronze 에서 다시 Silver 재생성
3. 협업 시 구조
	1. Data Engineer : Bronze / Silver
	2. Data Analyst : Gold
	3. ML Engineer : Silver / Gold

- Bronze Layer
> 원본 데이터 보관 (Raw Data)

1. 원본 그대로 저장
2. 최소 변환
3. 데이터 백업 역할
4. 형식 : json, csv, parquet (ex. `bronze/taxi/table.parquet`)
5. 목적
	1. 데이터 유실 방지
	2. 재처리 가능

- Silver Layer
>정제된 데이터
>대부분의 작업이 이루어지는 Layer

1. 결측치 처리
2. 이상치 처리
3. 데이터 타입 정리
4. 스키마 정리
5. 중복 제거

- Gold Layer
>비즈니스 분석 데이터
>대부분 집계 데이터를 만드는 곳
>Dashboard, BI, ML 모델에 사용되는 Layer

월별 매출, 지역별 택시 수요, 고객 분석 등

# 1️⃣. Data Lake 구축

## 1.1 목표

> AWS에 Data 저장소 구축

## 1.2 작업

1. S3 Bucket 생성
2. IAM Role 생성
3. IAM Policy 설정
4. DataBricks Credential 생성
5. External Location 연결

## 1.3 구조

```
AWS S3
   ↓
External Location
   ↓
Databricks
```


# 2️⃣. Raw Data Ingestion

## 2.1 목표

> Raw 데이터를 DataBricks 테이블로 로드

## 2.2 데이터 구조 확인

- 구조

```sql
describe default.table
```

|col_name|data_type|comment|
|---|---|---|
|VendorID|bigint|null|
|tpep_pickup_datetime|timestamp|null|
|tpep_dropoff_datetime|timestamp|null|
|passenger_count|double|null|
|trip_distance|double|null|
|RatecodeID|double|null|
|store_and_fwd_flag|string|null|
|PULocationID|bigint|null|
|DOLocationID|bigint|null|
|payment_type|bigint|null|
|fare_amount|double|null|
|extra|double|null|
|mta_tax|double|null|
|tip_amount|double|null|
|tolls_amount|double|null|
|improvement_surcharge|double|null|
|total_amount|double|null|
|congestion_surcharge|double|null|
|airport_fee|double|null|
|_rescued_data|string|null|
## 2.3 데이터 확인

```SQL
SELECT *
FROM default.table
LIMIT 10
```

| VendorID | tpep_pickup_datetime          | tpep_dropoff_datetime         | passenger_count | trip_distance | RatecodeID | store_and_fwd_flag | PULocationID | DOLocationID | payment_type | fare_amount | extra | mta_tax | tip_amount | tolls_amount | improvement_surcharge | total_amount | congestion_surcharge | airport_fee | _rescued_data |
| -------- | ----------------------------- | ----------------------------- | --------------- | ------------- | ---------- | ------------------ | ------------ | ------------ | ------------ | ----------- | ----- | ------- | ---------- | ------------ | --------------------- | ------------ | -------------------- | ----------- | ------------- |
| 2        | 2023-01-01T00:32:10.000+00:00 | 2023-01-01T00:40:36.000+00:00 | 1.0             | 0.97          | 1.0        | N                  | 161          | 141          | 2            | 9.3         | 1.0   | 0.5     | 0.0        | 0.0          | 1.0                   | 14.3         | 2.5                  | 0.0         | null          |
| 2        | 2023-01-01T00:55:08.000+00:00 | 2023-01-01T01:01:27.000+00:00 | 1.0             | 1.1           | 1.0        | N                  | 43           | 237          | 1            | 7.9         | 1.0   | 0.5     | 4.0        | 0.0          | 1.0                   | 16.9         | 2.5                  | 0.0         | null          |
| 2        | 2023-01-01T00:25:04.000+00:00 | 2023-01-01T00:37:49.000+00:00 | 1.0             | 2.51          | 1.0        | N                  | 48           | 238          | 1            | 14.9        | 1.0   | 0.5     | 15.0       | 0.0          | 1.0                   | 34.9         | 2.5                  | 0.0         | null          |
| 1        | 2023-01-01T00:03:48.000+00:00 | 2023-01-01T00:13:25.000+00:00 | 0.0             | 1.9           | 1.0        | N                  | 138          | 7            | 1            | 12.1        | 7.25  | 0.5     | 0.0        | 0.0          | 1.0                   | 20.85        | 0.0                  | 1.25        | null          |
| 2        | 2023-01-01T00:10:29.000+00:00 | 2023-01-01T00:21:19.000+00:00 | 1.0             | 1.43          | 1.0        | N                  | 107          | 79           | 1            | 11.4        | 1.0   | 0.5     | 3.28       | 0.0          | 1.0                   | 19.68        | 2.5                  | 0.0         | null          |
| 2        | 2023-01-01T00:50:34.000+00:00 | 2023-01-01T01:02:52.000+00:00 | 1.0             | 1.84          | 1.0        | N                  | 161          | 137          | 1            | 12.8        | 1.0   | 0.5     | 10.0       | 0.0          | 1.0                   | 27.8         | 2.5                  | 0.0         | null          |
| 2        | 2023-01-01T00:09:22.000+00:00 | 2023-01-01T00:19:49.000+00:00 | 1.0             | 1.66          | 1.0        | N                  | 239          | 143          | 1            | 12.1        | 1.0   | 0.5     | 3.42       | 0.0          | 1.0                   | 20.52        | 2.5                  | 0.0         | null          |
| 2        | 2023-01-01T00:27:12.000+00:00 | 2023-01-01T00:49:56.000+00:00 | 1.0             | 11.7          | 1.0        | N                  | 142          | 200          | 1            | 45.7        | 1.0   | 0.5     | 10.74      | 3.0          | 1.0                   | 64.44        | 2.5                  | 0.0         | null          |
| 2        | 2023-01-01T00:21:44.000+00:00 | 2023-01-01T00:36:40.000+00:00 | 1.0             | 2.95          | 1.0        | N                  | 164          | 236          | 1            | 17.7        | 1.0   | 0.5     | 5.68       | 0.0          | 1.0                   | 28.38        | 2.5                  | 0.0         | null          |
| 2        | 2023-01-01T00:39:42.000+00:00 | 2023-01-01T00:50:36.000+00:00 | 1.0             | 3.01          | 1.0        | N                  | 141          | 107          | 2            | 14.9        | 1.0   | 0.5     | 0.0        | 0.0          | 1.0                   | 19.9         | 2.5                  | 0.0         | null          |
| ...      |                               |                               |                 |               |            |                    |              |              |              |             |       |         |            |              |                       |              |                      |             |               |


# 3️⃣. Data Profiling (EDA)

## 3.1 목표

> 데이터 구조 이해

## 3.2 분석 과제

1. 평균 요금
2. 평균 거리

```sql
SELECT
    AVG(total_amount) avg_fare,
    AVG(trip_distance) avg_distance
FROM default.table
```

| avg_total          | avg_distance      |
| ------------------ | ----------------- |
| 27.020383107156004 | 3.847342030660308 |

3. 결제 방식

```SQL
SELECT
  round(AVG(TOTAL_AMOUNT), 2) AS avg_total,
  payment_type
FROM default.table
GROUP BY payment_type
ORDER BY payment_type
LIMIT 10
```

| avg_total | payment_type |
| --------- | ------------ |
| 29.13     | 0            |
| 28.3      | 1            |
| 23.03     | 2            |
| 10.51     | 3            |
| 2.55      | 4            |

4. 시간대 수요

```SQL
SELECT
    HOUR(tpep_pickup_datetime) AS pickup_hour,
    COUNT(*) AS trip_count,
    RANK() OVER (ORDER BY COUNT(*) DESC) AS rank
FROM default.table
GROUP BY pickup_hour
ORDER BY pickup_hour
```

|pickup_hour|trip_count|rank|
|---|---|---|
|0|84969|18|
|1|59799|19|
|2|42040|21|
|3|27438|22|
|4|17835|24|
|5|18011|23|
|6|43860|20|
|7|86877|17|
|8|116865|15|
|9|131111|14|
|10|143666|13|
|11|154157|11|
|12|169858|8|
|13|178739|7|
|14|191604|6|
|15|196424|3|
|16|195977|4|
|17|209493|2|
|18|215889|1|
|19|192801|5|
|20|165862|9|
|21|161548|10|
|22|147415|12|
|23|114528|16|

# 4️⃣. Data Preprocessing

## 4.1 목표

> Raw 데이터를 전처리 과정으로 분석 용이하게 변환

## 4.2 전처리 과제

1. 이상값 제거

```sql
select
  *,
  case
    when trip_distance <= 0 then 'trip_distance'
    when total_amount <= 0 then 'total_amount'
    when passenger_count <= 0 then 'passenger_count'
  end as outlier_type
from default.table
where trip_distance <= 0
  or total_amount <= 0
  or passenger_count <= 0
limit 10
```

| VendorID | tpep_pickup_datetime          | tpep_dropoff_datetime         | passenger_count | trip_distance | RatecodeID | store_and_fwd_flag | PULocationID | DOLocationID | payment_type | fare_amount | extra | mta_tax | tip_amount | tolls_amount | improvement_surcharge | total_amount | congestion_surcharge | airport_fee | _rescued_data | outlier_type    |
| -------- | ----------------------------- | ----------------------------- | --------------- | ------------- | ---------- | ------------------ | ------------ | ------------ | ------------ | ----------- | ----- | ------- | ---------- | ------------ | --------------------- | ------------ | -------------------- | ----------- | ------------- | --------------- |
| 1        | 2023-01-01T00:03:48.000+00:00 | 2023-01-01T00:13:25.000+00:00 | 0               | 1.9           | 1          | N                  | 138          | 7            | 1            | 12.1        | 7.25  | 0.5     | 0          | 0            | 1                     | 20.85        | 0                    | 1.25        | null          | passenger_count |
| 2        | 2023-01-01T00:28:29.000+00:00 | 2023-01-01T00:31:03.000+00:00 | 1               | 0.42          | 1          | N                  | 233          | 229          | 4            | -5.1        | -1    | -0.5    | 0          | 0            | -1                    | -10.1        | -2.5                 | 0           | null          | total_amount    |
| 2        | 2023-01-01T00:20:18.000+00:00 | 2023-01-01T00:27:56.000+00:00 | 2               | 1.19          | 1          | N                  | 142          | 50           | 4            | -9.3        | -1    | -0.5    | 0          | 0            | -1                    | -14.3        | -2.5                 | 0           | null          | total_amount    |
| 2        | 2023-01-01T00:39:02.000+00:00 | 2023-01-01T00:46:03.000+00:00 | 1               | 0             | 1          | N                  | 137          | 162          | 1            | 7.9         | 1     | 0.5     | 3.22       | 0            | 1                     | 16.12        | 2.5                  | 0           | null          | trip_distance   |
| 2        | 2023-01-01T00:47:29.000+00:00 | 2023-01-01T00:55:49.000+00:00 | 1               | 0             | 1          | N                  | 233          | 141          | 1            | 8.6         | 1     | 0.5     | 2.72       | 0            | 1                     | 16.32        | 2.5                  | 0           | null          | trip_distance   |
| 2        | 2023-01-01T00:59:24.000+00:00 | 2023-01-01T01:14:26.000+00:00 | 1               | 0             | 1          | N                  | 141          | 193          | 2            | 13.5        | 1     | 0.5     | 0          | 0            | 1                     | 18.5         | 2.5                  | 0           | null          | trip_distance   |
| 2        | 2023-01-01T00:52:22.000+00:00 | 2023-01-01T01:14:03.000+00:00 | 1               | 4.89          | 1          | N                  | 238          | 167          | 4            | -25.4       | -1    | -0.5    | 0          | 0            | -1                    | -30.4        | -2.5                 | 0           | null          | total_amount    |
| 1        | 2023-01-01T00:57:44.000+00:00 | 2023-01-01T00:57:59.000+00:00 | 1               | 0             | 1          | N                  | 137          | 137          | 3            | 3           | 3.5   | 0.5     | 0          | 0            | 1                     | 8            | 2.5                  | 0           | null          | trip_distance   |
| 2        | 2023-01-01T00:28:04.000+00:00 | 2023-01-01T00:28:35.000+00:00 | 1               | 0             | 2          | N                  | 142          | 142          | 2            | 70          | 0     | 0.5     | 0          | 0            | 1                     | 74           | 2.5                  | 0           | null          | trip_distance   |
| 2        | 2023-01-01T00:37:17.000+00:00 | 2023-01-01T00:38:51.000+00:00 | 1               | 0             | 5          | N                  | 255          | 264          | 1            | 40          | 0     | 0       | 8.2        | 0            | 1                     | 49.2         | 0                    | 0           | null          | trip_distance   |
| ...      |                               |                               |                 |               |            |                    |              |              |              |             |       |         |            |              |                       |              |                      |             |               |                 |

- count

```sql
select
  sum(
    case when trip_distance <=0 then 1 else 0 end
  ) as distance_error,
  sum(
    case when total_amount <=0 then 1 else 0 end
  ) as fare_error,
  sum(
    case when passenger_count <=0 then 1 else 0 end
  ) as passenger_error
from default.table
```

|distance_error|fare_error|passenger_error|
|---|---|---|
|45862|25772|51164|

3. N/A 제거

```sql
select
  count(*) total_rows,
  count(tpep_dropoff_datetime) dropoff_not_null,
  count(tpep_pickup_datetime) pickup_not_null,
  count(passenger_count) passenger_not_null,
  count(trip_distance) distance_not_null
from default.table
```

|total_rows|dropoff_not_null|pickup_not_null|passenger_not_null|distance_not_null|
|---|---|---|---|---|
|3066766|3066766|3066766|2995023|3066766|

4. New Table (이상치, 결측치 제거)

```sql
create or replace table default.table_cleansing as
select * from default.table
where
  -- remove na
  tpep_dropoff_datetime is not null
and tpep_pickup_datetime is not null
and passenger_count is not null
and trip_distance is not null
and ratecodeid is not null
and store_and_fwd_flag is not null
and payment_type is not null
and fare_amount is not null
and extra is not null
and mta_tax is not null
and tip_amount is not null
and tolls_amount is not null
and improvement_surcharge is not null
and total_amount is not null
and congestion_surcharge is not null
and airport_fee is not null
  -- remove outlier
and trip_distance > 0
and passenger_count > 0
and total_amount >= 0
and fare_amount >= 0
```

5. check

```sql
select
  (select count(*) from default.table) as cnt_table,
  (select count(*) from default.table_cleansing) as cnt_table_cleansing
```

|cnt_table|cnt_table_cleansing|
|---|---|
|3066766|2884568|

# 5️⃣. Analytics Table

## 5.1 목표

> 분석용 테이블 생성

## 5.2 분석 테이블 과제

1. 시간별 수요

```sql
CREATE TABLE default.taxi_gold_hourly AS
SELECT
    pickup_hour,
    COUNT(*) AS trip_count,
    SUM(total_amount) AS revenue
FROM default.taxi_silver
GROUP BY pickup_hour
```


2. 지역별 수요

```sql
CREATE TABLE default.taxi_gold_location AS
SELECT
    PULocationID,
    COUNT(*) AS trip_count
FROM default.taxi_silver
GROUP BY PULocationID
```
## 5.3 구조

```
default
 ├ table
 ├ taxi_silver
 ├ taxi_gold_hourly
 └ taxi_gold_location
```


# 6️⃣. Parquet → Delta Lake 변환

## 6.1 목표

> Delta Lake 기능 사용

```sql
CREATE TABLE default.taxi_delta
USING DELTA
AS SELECT * FROM default.taxi_silver
```

## 6.2 Delta Lake 사용 이유

1. Time Travel
2. Update
3. Delete
4. Merge


# 7️⃣. Delta Lake Optimization

## 7.1 파일 압축

```sql
OPTIMIZE default.taxi_delta
```

## 7.2 Z-ORDER

```sql
OPTIMIZE default.taxi_delta
ZORDER BY (PULocationID)
```


# 8️⃣. Data Visualization

## 8.1 그래프

1. 시간대별 수요
2. 거리 대 요금 비교
3. 결제 방식


# 9️⃣. 머신러닝 모델

## 9.1 목표

> 택시 운임 요금 예측

## 9.2 Feature

1. `trip_distance`
2. `passenger_count`
3. `pickup_hour`

## 9.3 Target

- `total_amount`

# 1️⃣0️⃣. ML Pipeline

## 10.1 단계

```
Feature Engineering
↓
Model Training
↓
Model Evaluation
↓
Prediction
```


# 1️⃣1️⃣. Streaming

- Auto Loader 사용

```
S3 New Data
 ↓
Databricks Streaming
 ↓
Delta Table
```


# 1️⃣2️⃣. 전체 아키텍처

```
AWS S3
   ↓
External Location
   ↓
Bronze Table
   ↓
Silver Table
   ↓
Gold Table
   ↓
Delta Lake
   ↓
ML Model
```