import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

// WebSocket 이벤트 이름 정의 (Python 코드와 일치)
const QSAR_DESC_CALC_PROGRESS_EVENT = "qsar-desc-calc-progress";

//노드 지정
const TARGET_NODES = Object.freeze([
	//Data Loader
    "Data_Loader_Regression",
	"Data_Loader_Classification",
	//Standardization
    "Standardization_Regression",
	"Standardization_Classification",
    "Load_and_Standardize_Regression",
	"Load_and_Standardize_Classification",
	//Descriptor Calculations
    "Descriptor_Calculations_Regression",
    "Descriptor_Calculations_Classification",
	//Descriptor Preprocessing
    "Replace_inf_with_nan_Regression",
	"Replace_inf_with_nan_Classification",
    "Remove_high_nan_compounds_Regression",
	"Remove_high_nan_compounds_Classification",
    "Remove_high_nan_descriptors_Regression",
	"Remove_high_nan_descriptors_Classification",
    "Impute_missing_values_Regression",
	"Impute_missing_values_Classification",
    "Descriptor_preprocessing_Regression",
	"Descriptor_preprocessing_Classification",
	//Descriptor Optimization
    "Remove_Low_Variance_Features_Regression",
	"Remove_Low_Variance_Features_Classification",
    "Remove_High_Correlation_Features_Regression",
	"Remove_High_Correlation_Features_Classification",
    "Descriptor_Optimization_Regression",
	"Descriptor_Optimization_Classification",
	//FEATURE SELECTION
    "Feature_Selection_Regression",
	"Feature_Selection_Classification",
	//GRID SEARCH HYPERPARAMETER
	"Hyperparameter_Grid_Search_Regression",
	"Hyperparameter_Grid_Search_Classification",
	//MODEL VALIDATION
	"Model_Validation_Regression",
	"Model_Validation_Classification",
]);

// 노드 크기 관련 상수
const MIN_NODE_WIDTH = 300;   // 최소 노드 너비 (픽셀)
const MAX_NODE_WIDTH = 800;   // 최대 노드 너비 (픽셀)
const CHAR_WIDTH = 7.3;       // 평균 문자 너비 (픽셀) - 모노스페이스 폰트 기준
const LINE_HEIGHT = 15;       // 한 줄 높이 (픽셀)
const MIN_NODE_HEIGHT = 100;  // 최소 노드 높이 (픽셀)
const MAX_NODE_HEIGHT = 600;  // 최대 노드 높이 (픽셀)
const BASE_HEIGHT = 70;       // 기본 노드 높이 (텍스트 외 영역, 헤더 등)
const HORIZONTAL_PADDING = 1; // 노드 좌우 추가 여백 (픽셀)
const TOP_PADDING = 20;       // 노드 상단 여백 (픽셀)
const BOTTOM_PADDING = 20;    // 텍스트 위젯 아래 추가 여백 (픽셀)
const WIDGET_PADDING = 16;    // 위젯 내부 패딩 (padding: 8px * 2)

// 노드 업데이트 이벤트를 처리하기 위한 전역 리스너
document.addEventListener("update_node", function(event) {
    const { detail } = event;
    if (!detail || !detail.node) return;
    
    // 노드 ID에 해당하는 노드 객체 찾기
    const node = app.graph.getNodeById(detail.node);
    if (!node) return;
    
    // 노드 업데이트 처리
    if (typeof node.onNodeUpdateMessage === "function") {
        node.onNodeUpdateMessage(detail);
    }
});

app.registerExtension({
    name: "ComfyQSAR_TEXT",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (TARGET_NODES.includes(nodeData.name)) {
            // 텍스트 위젯 찾기 또는 생성 (이 함수는 노드 인스턴스(this)를 받음)
            function findOrCreateTextWidget(node) {
                let textWidget = node.widgets?.find(w => w.name === "text2");
                if (!textWidget) {
                    try {
                        // 노드에 위젯이 없으면 text2 위젯 생성
                        textWidget = ComfyWidgets["STRING"](node, "text2", ["STRING", { multiline: true }], app).widget;
                        // console.log(`[${node.title}] Created text2 widget.`); // 디버깅용
                    } catch (error) {
                        console.error(`[${node.title}] Error creating text2 widget:`, error);
                        return null;
                    }
                }
                // 초기에는 숨겨둠 (populate에서 필요시 표시)
                hideTextWidget(node, textWidget);
                return textWidget;
            }

            // 텍스트 위젯 숨기기
            function hideTextWidget(node, textWidget) {
                if (!textWidget) textWidget = node.widgets?.find(w => w.name === "text2");
                if (textWidget && textWidget.inputEl) {
                    textWidget.inputEl.style.display = "none";
                    // 애니메이션 중지 및 초기화
                    if (node.animationFrameId) cancelAnimationFrame(node.animationFrameId);
                    node.animationFrameId = null;
                    node.currentVisualProgress = 0;
                    node.currentTargetProgress = 0;
                    // 상태 초기화 (메시지만)
                    node.currentMessage = "";
                    // 스타일 초기화
                    textWidget.inputEl.style.background = "white";
                    textWidget.inputEl.style.transition = "none";
                }
                 // 위젯이 숨겨질 때 노드 크기 재계산 (선택적)
                 // node.computeSize();
                 // node.setDirtyCanvas(true, true);
            }

            // 텍스트 위젯 표시
            function showTextWidget(node, textWidget) {
                if (!textWidget) textWidget = node.widgets?.find(w => w.name === "text2");
                if (textWidget && textWidget.inputEl) {
                    textWidget.inputEl.style.display = "block"; // block으로 변경
                    // 라벨은 계속 숨김
                    // if (textWidget.labelEl) textWidget.labelEl.style.display = "none";
                }
            }

            // 텍스트 위젯 내용 및 크기 업데이트 (애니메이션 시작/관리 포함 안 함)
            function updateTextAndSize(messageText) {
                 if (!this.widgets) { return; }
                 let textWidget = this.widgets.find(w => w.name === "text2");
                 if (!textWidget) return;

                 const v = Array.isArray(messageText) ? [...messageText] : [messageText];
                 if (v.length === 0 || (v.length === 1 && (v[0] === null || v[0] === undefined || v[0] === ""))) {
                     hideTextWidget(this, textWidget);
                     this.computeSize(); // 크기 재계산 추가
                     this.setDirtyCanvas(true, true);
                     return;
                 }
                 if (v.length > 0 && !v[0]) { v.shift(); }

                 const combinedText = v.join(''); // 메시지만 사용

                 // 텍스트 위젯 값 설정
                 textWidget.value = combinedText;
                 textWidget.inputEl.readOnly = true;
                 textWidget.inputEl.style.opacity = 1.0;

                 // 크기 계산
                 const lines = combinedText.split('\n');
                 const lineCount = lines.length;
                 const longestLine = lines.reduce((longest, line) => line.length > longest.length ? line : longest, "");
                 const calculatedWidth = Math.min(MAX_NODE_WIDTH, Math.max(MIN_NODE_WIDTH, Math.ceil(longestLine.length * CHAR_WIDTH) + WIDGET_PADDING));
                 const contentHeight = (lineCount * LINE_HEIGHT) + WIDGET_PADDING;
                 const requiredNodeHeight = BASE_HEIGHT + contentHeight + TOP_PADDING + BOTTOM_PADDING;
                 const finalNodeHeight = Math.min(MAX_NODE_HEIGHT, Math.max(MIN_NODE_HEIGHT, requiredNodeHeight));
                 const availableWidgetHeight = Math.max(20, finalNodeHeight - BASE_HEIGHT - TOP_PADDING - BOTTOM_PADDING);

                 // 스타일 설정 (배경 흰색 고정, 전환 없음)
                 textWidget.inputEl.style.width = "100%";
                 textWidget.inputEl.style.height = `${availableWidgetHeight}px`;
                 textWidget.inputEl.style.whiteSpace = "pre-wrap";
                 textWidget.inputEl.style.overflow = "auto";
                 textWidget.inputEl.style.wordBreak = "break-word";
                 textWidget.inputEl.style.textAlign = "left";
                 textWidget.inputEl.style.padding = "8px";
                 textWidget.inputEl.style.fontFamily = "monospace";
                 textWidget.inputEl.style.fontSize = "10px";
                 textWidget.inputEl.style.fontWeight = "normal";
                 textWidget.inputEl.style.color = "black";
                 textWidget.inputEl.style.borderRadius = "6px";
                 textWidget.inputEl.style.border = "1px solid #cccccc";
                 textWidget.inputEl.style.background = "white"; // 배경 흰색 고정
                 textWidget.inputEl.style.transition = "none"; // 전환 없음

                 showTextWidget(this, textWidget);
                 this.setSize([calculatedWidth, finalNodeHeight]);
                 this.setDirtyCanvas(true, true);
            }

            // 노드 생성 시 처리 (onNodeCreated 수정)
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                findOrCreateTextWidget(this); // 위젯 생성 및 초기 숨김

                // 상태 변수 초기화 (애니메이션 포함)
                this.currentMessage = "";
                this.currentVisualProgress = 0;
                this.currentTargetProgress = 0;
                this.animationFrameId = null;

                const textWidget = this.widgets?.find(w => w.name === "text2");

                if (textWidget) {
                     const handleProgressUpdate = (event) => {
                         // 필터링: node id가 없거나 현재 노드와 일치하지 않으면 무시
                         const sourceId = event.detail.node || event.detail.sid;
                         if (!sourceId || sourceId !== this.id) return;

                        const progress = event.detail.progress !== undefined ? event.detail.progress : null; // 진행률 받음
                        const text = event.detail.text; // 메시지 받음

                        if (text !== undefined && text !== null) {
                             // 1. 현재 메시지 업데이트
                             this.currentMessage = Array.isArray(text) ? text.join('') : text;
                             // 2. 텍스트 및 크기 업데이트
                             updateTextAndSize.call(this, this.currentMessage);
                        }

                        // 2. 목표 진행률 업데이트 및 애니메이션 시작/계속
                        if (progress !== null && progress >= 0 && progress <= 100) {
                             this.currentTargetProgress = Math.round(progress);
                             // 애니메이션 시작 또는 계속
                             if (this.animationFrameId === null) {
                                 this.animationFrameId = requestAnimationFrame(animateProgress.bind(this));
                             }
                        }
                     };
                     this.handleProgressUpdateRef = handleProgressUpdate;
                     app.api.addEventListener(QSAR_DESC_CALC_PROGRESS_EVENT, this.handleProgressUpdateRef);
                } else {
                    console.warn(`[${this.title}] Could not find text2 widget on nodeCreated to add listener.`);
                }

                const onRemoved = this.onRemoved;
                this.onRemoved = () => {
                    // 리스너 제거
                    if (this.handleProgressUpdateRef) {
                        app.api.removeEventListener(QSAR_DESC_CALC_PROGRESS_EVENT, this.handleProgressUpdateRef);
                        this.handleProgressUpdateRef = null;
                    }
                    // 애니메이션 중지
                    if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
                    this.animationFrameId = null;
                    onRemoved?.apply(this, arguments);
                };
                return result;
            };

            // 노드 실행 완료 시 처리 (onExecuted 수정)
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                // 리스너 제거
                if (this.handleProgressUpdateRef) {
                    app.api.removeEventListener(QSAR_DESC_CALC_PROGRESS_EVENT, this.handleProgressUpdateRef);
                    this.handleProgressUpdateRef = null;
                }
                // 애니메이션 중지
                if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
                this.animationFrameId = null;
                // 상태 설정 (완료 시 시각적 진행률 100)
                this.currentVisualProgress = 100;
                this.currentTargetProgress = 100;
                this.currentMessage = (message && message.text) ? (Array.isArray(message.text) ? message.text.join('') : message.text) : "";

                 // 최종 결과 표시 (진행 정보 없음)
                 if (this.currentMessage) {
                      updateTextAndSize.call(this, this.currentMessage);
                      // 최종 배경 흰색 설정
                      const textWidget = this.widgets?.find(w => w.name === "text2");
                      if(textWidget) textWidget.inputEl.style.background = "white";
                 } else {
                      hideTextWidget(this); // 메시지 없으면 숨김
                 }
            };

            // 노드 설정 시 처리 (onConfigure 수정)
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                // 상태 초기화 (애니메이션 포함)
                if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
                this.currentVisualProgress = 0;
                this.currentTargetProgress = 0;
                this.animationFrameId = null;
                this.currentMessage = "";

                const textWidget = findOrCreateTextWidget(this);

                setTimeout(() => {
                    let outputText = null;
                    // 이전 결과 텍스트 찾는 로직 (진행 정보 제거 필요 없음)
                    if (this.widgets_values && Array.isArray(this.widgets_values)) {
                        for(let i = this.widgets_values.length - 1; i >= 0; i--) {
                             const val = this.widgets_values[i];
                             if (typeof val === 'string' && val.includes('\n') && (val.includes('🔹') || val.includes('❌'))) {
                                 outputText = val; // 진행 정보 제거 불필요
                                 break;
                             }
                         }
                    }

                    if (outputText) {
                        this.currentMessage = outputText; // 복원된 메시지 저장
                        // 이전 결과 로드
                        updateTextAndSize.call(this, this.currentMessage);
                        // 배경 흰색 설정
                        if(textWidget) textWidget.inputEl.style.background = "white";
                    } else {
                         hideTextWidget(this, textWidget); // 결과 없으면 숨김
                    }
                }, 150);
            };

            // --- 애니메이션 함수 (다시 추가) ---
            function animateProgress() {
                if (!this.widgets) return; // 노드 제거 시 중지
                const textWidget = this.widgets.find(w => w.name === "text2");
                if (!textWidget || !textWidget.inputEl) {
                    this.animationFrameId = null;
                    return; // 위젯 없으면 중지
                }

                const target = this.currentTargetProgress;
                const current = this.currentVisualProgress;
                const diff = target - current;

                // 목표 도달 시 또는 매우 근접 시 중지
                if (Math.abs(diff) < 0.1) {
                    this.currentVisualProgress = target;
                    // 최종 배경 업데이트
                    textWidget.inputEl.style.background = `linear-gradient(to right, #64b5f6 ${target}%, #e3f2fd ${target}%)`;
                    this.animationFrameId = null; // 애니메이션 ID 초기화
                    return;
                }

                // 현재 진행률을 목표치에 가깝게 이동 (Easing 효과)
                this.currentVisualProgress += diff * 0.08; // 부드럽게

                // 배경 업데이트
                textWidget.inputEl.style.background = `linear-gradient(to right, #64b5f6 ${this.currentVisualProgress}%, #e3f2fd ${this.currentVisualProgress}%)`;

                // 다음 프레임 요청
                this.animationFrameId = requestAnimationFrame(animateProgress.bind(this));
            }
            // ------------------------------
        }
    }
});