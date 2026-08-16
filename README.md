# Semiconductor AI Hackathon Bootcamp

반도체 공정 문제를 **데이터 감사 → 기준모델 → What-if 시뮬레이션 → 의사결정 → GitHub Pages → 3분 발표**로 연결하는 단기 특강 준비 저장소입니다.

도구 흐름은 **Claude·GPT·Gemini 중 하나로 계획·분석 → 같은 로컬 테스트와 Holdout 검증 → GitHub Push → GitHub Pages 포트폴리오**입니다.

> 교육용 합성 데이터만 사용합니다. 특정 회사의 공식 교육과정·실제 공정 조건·기술 노드·내부 Spec을 나타내지 않습니다.

## 운영 구조

- 1차: 2026-08-14, 오프라인 2시간
- 자율 프로젝트: 2026-08-20 이후, 같은 8단계를 개인 데이터로 반복
- 대상: 반도체 소자·R&D 공정·양산기술·설비기술 지원자 10명
- 결과물: 개인 저장소, 작동형 MVP, Live Page, AI 활용기록, 3분 발표

## 첫날 성공 기준

수업이 끝날 때 모든 수강생이 다음 다섯 가지를 이해하고, 최소 한 단계는 직접 수행해야 합니다.

1. 사용자·데이터·결정을 포함한 문제정의
2. 선택한 AI를 이용한 근거 조사와 근거카드
3. Claude·GPT·Gemini 공통 프롬프트로 저장소를 읽고 계획하는 과정
4. AI 결과를 데이터·전공 원리·반례로 검증하는 방법
5. Git diff·Commit·Pages로 과정의 증거를 남기는 방법

개인 Fork·Commit·Pages는 사전 준비 상태에 따라 수업 중 또는 수업 후 완성합니다.

## 🚀 추가된 프로젝트 산출물 (Universal ML Pipeline)

본 저장소에는 **모든 반도체 공정에 적용 가능한 범용 머신러닝 파이프라인**과 **포토 공정(01_photo) 맞춤형 예측 모델**이 새롭게 추가되었습니다. 

- **분석 보고서**: [`analysis_index.html`](analysis_index.html) (EDA, 다중공선성 VIF 검증, 그룹별 독립 회귀 분석, 클래스 불균형에 대응한 분류 모델 성능 요약)
- **범용 에이전트 스킬**: [`.agents/skills/semiconductor-process-ml-pipeline/SKILL.md`](.agents/skills/semiconductor-process-ml-pipeline/SKILL.md) (어떤 공정 데이터든 스스로 문맥을 읽어내어 VIF 검사, 그룹핑, 모델 분리, 시각화를 수행하는 마스터 템플릿)
- **포토 공정 특화 스킬**: [`.agents/skills/photo-process-ml-pipeline/SKILL.md`](.agents/skills/photo-process-ml-pipeline/SKILL.md) (PR Tone 분리, Domain Shift 방지 등 도메인 지식 반영)
- **제출용 결과물**: `A_holdout_predictions.csv`, `B_holdout_predictions.csv`

## 저장소 안내

- [`index.html`](index.html): 1차 강의용 반응형 GitHub Pages 자료
- [`preclass_setup.html`](preclass_setup.html): 계정·프로그램·패키지·토큰을 한 번에 확인하는 Windows 사전 준비 페이지
- [`COURSE_PLAN.md`](COURSE_PLAN.md): 과정 범위·일정·완료 기준
- [`STATUS.md`](STATUS.md): 현재 준비상태와 남은 우선순위
- [`instructor/`](instructor/): 1·2차 진행표, 사전점검
- [`student/00_PROBLEM_DISCOVERY_QUESTIONNAIRE.md`](student/00_PROBLEM_DISCOVERY_QUESTIONNAIRE.md): 문제→전공→Governing equation·도메인 제약→상관·대안가설→의사결정으로 좁히는 1강 질문지
- [`student/00_PRECLASS_SETUP_2026-08-14.md`](student/00_PRECLASS_SETUP_2026-08-14.md): 수강생에게 전달할 간단한 설치 체크리스트
- [`templates/UNIVERSAL_AI_PROJECT_PROMPT.md`](templates/UNIVERSAL_AI_PROJECT_PROMPT.md): Claude·GPT·Gemini 공통 프로젝트 프롬프트
- [`templates/COMPACT_AI_PROMPTS.md`](templates/COMPACT_AI_PROMPTS.md): 수업 중 계획·구현·검수 3회용 압축 프롬프트
- [`student/AI_QUOTA_SAFETY_PLAN.md`](student/AI_QUOTA_SAFETY_PLAN.md): Claude·GPT 한도 보호와 Gemini·로컬 대체 경로
- [`student/01_FIRST_CLASS_HANDS_ON_MANUAL.md`](student/01_FIRST_CLASS_HANDS_ON_MANUAL.md): 설치부터 첫 Pages 배포까지 수강생 실습서
- [`instructor/01_TELEGRAM_DELIVERY_SCRIPT.md`](instructor/01_TELEGRAM_DELIVERY_SCRIPT.md): 20명 동시 진행용 Telegram 메시지 대본
- [`instructor/SESSION1_SMOOTH_FLOW.md`](instructor/SESSION1_SMOOTH_FLOW.md): 통계·시각화·AI·배포가 이어지는 120분 강사 진행표
- [`instructor/SESSION1_AGENT_WORKFLOW_10_BEGINNERS.md`](instructor/SESSION1_AGENT_WORKFLOW_10_BEGINNERS.md): GitHub 미가입자가 포함된 10명 초급반용 Agent 시연·페어 실습·시간 부족 대응안
- [`templates/PLAN.md`](templates/PLAN.md): 현재 문제의 목표·전공지식·가설·실행·검증 기준 작성 틀
- [`templates/SKILL.md`](templates/SKILL.md): 여러 프로젝트에서 검증된 문제해결 절차를 재사용하기 위한 작성 틀
- [`student/02_TOPIC_AND_PROJECT_GUIDE.md`](student/02_TOPIC_AND_PROJECT_GUIDE.md): 주제선정·범위축소·7일 프로젝트 가이드
- [`student/03_INDIVIDUAL_ACHIEVEMENT_GUIDE.md`](student/03_INDIVIDUAL_ACHIEVEMENT_GUIDE.md): 설치→데이터카드→근거카드→주장감사→MVP→발표의 개인 성과 집중 가이드
- [`lessons/01_DATA_QUALITY_AND_VISUALIZATION.md`](lessons/01_DATA_QUALITY_AND_VISUALIZATION.md): 결측·이상치·편중 처리와 ggplot2·seaborn 그래프 선택 실습
- [`challenges/`](challenges/): 반도체 AI 문제 10종 카탈로그
- [`datasets/`](datasets/): 노이즈·결측·이상치·교란이 포함된 A/B 합성 데이터 20팩
- [`templates/`](templates/): 계획·AI 기록·발표·평가 양식
- [`demo/`](demo/): CMP 합성 데이터 기반 작동형 시연 예제

## 데모 실행

```bash
cd demo
python -m pip install -r requirements.txt
python src/build_demo.py
python -m http.server 8000 --directory docs
```

브라우저에서 <http://localhost:8000>을 열고 Down Force, 속도, Slurry, Pad Age, Pattern Density를 바꿔 결과가 갱신되는지 확인합니다.

## 보안 원칙

- 실제 Fab 데이터·장비 로그·내부 Spec·고객정보·개인정보를 업로드하지 않습니다.
- API Key·토큰·계정정보를 코드, Prompt, Screenshot, Commit에 남기지 않습니다.
- AI가 생성한 코드와 해석은 테스트·수치·원문 근거로 사람이 검증합니다.
- 상관관계를 인과관계로 표현하지 않습니다.

## 전체 데이터 재생성·검증

```bash
python tools/generate_datasets.py
python -m unittest discover -s tests -v
```

수강생 환경점검:

```bash
python tools/student_preflight.py
```
