import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

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
            // 텍스트 위젯 생성 또는 업데이트
            function populate(text, isProgressUpdate = false) {
                if (!this.widgets) {
                    return;
                }
                
                // 텍스트 확인 (배열 또는 문자열)
                const v = Array.isArray(text) ? [...text] : [text];
                
                // 빈 텍스트 처리
                if (v.length === 0 || (v.length === 1 && (!v[0] || v[0] === ""))) {
                    hideTextWidget(this);
                    // 노드 크기 초기화
                    this.setSize([this.size[0], this.computeSize()[1]]);
                    return;
                }
                
                // 배열에서 빈 첫 요소 제거
                if (v.length > 0 && !v[0]) {
                    v.shift();
                }
                
                // 텍스트를 합쳐서 하나의 문자열로 만듦
                const combinedText = v.join('');
                
                // 텍스트 위젯 찾거나 생성
                let textWidget = findOrCreateTextWidget(this);
                if (!textWidget) {
                    return;
                }
                
                // 위젯 값 설정
                textWidget.inputEl.readOnly = true;
                textWidget.inputEl.style.opacity = 1.0;
                textWidget.value = combinedText;
                
                // 텍스트 라인 분석
                const lines = combinedText.split('\n');
                const lineCount = lines.length;
                const longestLine = lines.reduce((longest, line) => 
                    line.length > longest.length ? line : longest, "");
                
                // 텍스트 길이에 따른 너비 계산
                const calculatedWidth = Math.min(
                    MAX_NODE_WIDTH, 
                    Math.max(MIN_NODE_WIDTH, Math.ceil(longestLine.length * CHAR_WIDTH) + HORIZONTAL_PADDING)
                );
                
                // 텍스트 내용에 필요한 높이 계산 (줄 수 * 줄 높이 + 위젯 내부 패딩)
                const contentHeight = (lineCount * LINE_HEIGHT) + WIDGET_PADDING;
                
                // 노드에 필요한 전체 높이 계산 (기본 높이 + 내용 높이 + 상단/하단 여백)
                const requiredNodeHeight = BASE_HEIGHT + contentHeight + TOP_PADDING + BOTTOM_PADDING;
                
                // 노드 높이 결정 (최소/최대 제한 적용)
                const finalNodeHeight = Math.min(
                    MAX_NODE_HEIGHT,
                    Math.max(MIN_NODE_HEIGHT, requiredNodeHeight)
                );

                // 텍스트 위젯 자체의 높이 계산 (노드 내에서 사용 가능한 높이, 하단 여백 제외)
                const availableWidgetHeight = Math.max(0, finalNodeHeight - BASE_HEIGHT - TOP_PADDING - BOTTOM_PADDING);
                
                // 텍스트 스타일 설정
                textWidget.inputEl.style.width = "100%";
                textWidget.inputEl.style.maxWidth = `${calculatedWidth}px`;
                textWidget.inputEl.style.height = `${availableWidgetHeight}px`; // 노드 높이에 맞춰 위젯 높이 설정
                textWidget.inputEl.style.whiteSpace = "pre-wrap"; // 줄바꿈 유지
                textWidget.inputEl.style.overflow = "auto"; // 내용이 위젯보다 크면 스크롤
                textWidget.inputEl.style.wordBreak = "break-word"; // 단어 내에서도 줄바꿈 허용
                textWidget.inputEl.style.textAlign = "left";
                textWidget.inputEl.style.padding = "8px";
                textWidget.inputEl.style.fontFamily = "monospace";
                textWidget.inputEl.style.fontSize = "10px";
                textWidget.inputEl.style.fontWeight = "normal";
                textWidget.inputEl.style.color = "black";
                textWidget.inputEl.style.backgroundColor = "white";
                textWidget.inputEl.style.borderRadius = "6px";
                textWidget.inputEl.style.border = "1px solid #cccccc"; // 얇은 회색 테두리 추가
                
                // 진행 상태 업데이트인 경우 테두리 스타일 변경
                if (isProgressUpdate) {
                    textWidget.inputEl.style.borderLeft = "3px solid #4a90e2";
                } else {
                    textWidget.inputEl.style.borderLeft = "1px solid #cccccc"; // 일반 테두리와 동일하게 변경
                }
                
                // 위젯 표시
                showTextWidget(this, textWidget);
                
                // 노드 크기 조정 (너비와 높이 모두 조정)
                this.setSize([calculatedWidth, finalNodeHeight]);
            }
            
            // 텍스트 위젯 찾기 또는 생성
            function findOrCreateTextWidget(node) {
                // 이미 존재하는 텍스트 위젯 찾기
                let textWidget = node.widgets.find(w => w.name === "text2");
                
                // 위젯이 없으면 새로 생성
                if (!textWidget) {
                    try {
                        // 초기 위젯 수 저장
                        if (!node.hasOwnProperty("initialWidgetsCount")) {
                            node.initialWidgetsCount = node.widgets.length;
                        }
                        
                        // 텍스트 위젯 생성
                        textWidget = ComfyWidgets["STRING"](node, "text2", ["STRING", { multiline: true }], app).widget;
                        
                        // 생성된 위젯 숨기기
                        hideTextWidget(node);
                    } catch (error) {
                        console.error("ComfyQSAR_TEXT: 텍스트 위젯 생성 중 오류", error);
                        return null;
                    }
                }
                
                return textWidget;
            }
            
            // 텍스트 위젯 숨기기
            function hideTextWidget(node) {
                const textWidget = node.widgets.find(w => w.name === "text2");
                if (textWidget) {
                    textWidget.inputEl.style.display = "none";
                    if (textWidget.labelEl) {
                        textWidget.labelEl.style.display = "none";
                    }
                }
            }
            
            // 텍스트 위젯 표시
            function showTextWidget(node, textWidget) {
                if (textWidget) {
                    textWidget.inputEl.style.display = "block";
                    if (textWidget.labelEl) {
                        textWidget.labelEl.style.display = "none"; // 라벨은 계속 숨김
                    }
                }
            }
            
            // 노드 업데이트 메시지 처리 함수 추가
            nodeType.prototype.onNodeUpdateMessage = function(info) {
                if (info && info.ui && info.ui.text) {
                    // 진행 중인지 완료인지 확인 (진행 중일 때는 "진행 중" 텍스트가 포함됨)
                    const isProgressUpdate = info.ui.text.includes("진행 중");
                    
                    // 텍스트 위젯 업데이트
                    populate.call(this, info.ui.text, isProgressUpdate);
                }
            };
            
            // 노드 생성 시 처리
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                
                // 위젯 생성 및 숨김
                findOrCreateTextWidget(this);
                hideTextWidget(this);
                
                return result;
            };
            
            // 노드 실행 시 결과 표시
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                // 메시지에 텍스트가 있는 경우에만 처리 (최종 결과로 표시)
                if (message && message.text) {
                    populate.call(this, message.text, false);
                }
            };
            
            // 노드 설정 시 처리 (워크플로우 로드/새로고침)
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                
                // 약간의 지연 후 처리
                setTimeout(() => {
                    let outputText = null;
                    if (this.widgets_values) {
                        // 위젯 값 배열에서 실제 출력 텍스트로 보이는 값을 찾음
                        // (개행 문자와 구분선 포함 여부로 판단)
                        for (const val of this.widgets_values) {
                            // 문자열이고, 줄바꿈과 '=' 문자를 모두 포함하는지 확인
                            if (typeof val === 'string' && val.includes('\n') && val.includes('=')) {
                                outputText = val;
                                break; // 첫 번째 매칭되는 값을 사용
                            }
                        }
                    }

                    if (outputText) {
                        // 식별된 이전 실행 결과가 있다면 표시
                        populate.call(this, [outputText], false);
                    } else {
                        // 저장된 결과가 없거나 형식에 맞지 않으면 위젯 숨김
                        hideTextWidget(this);
                    }
                }, 50); // 위젯 렌더링 후 실행되도록 지연
            };
        }
    }
});