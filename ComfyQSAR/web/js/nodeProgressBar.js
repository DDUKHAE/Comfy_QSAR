/**
 * Node Progress Bar 기능
 * 
 * 이 파일은 각 노드 내부에 진행 상태 표시줄을 추가하는 기능을 구현합니다.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_PROGRESS_STYLE = `
.qsar-node-progress {
  width: 100%;
  height: 6px;
  background-color: var(--comfy-input-bg);
  margin-top: 2px;
  margin-bottom: 2px;
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.qsar-node-progress-bar {
  position: absolute;
  height: 100%;
  background-color: green;
  transition: width 0.3s ease;
  width: 0;
  left: 0;
  top: 0;
}

.qsar-node-progress-text {
  font-size: 10px;
  color: var(--input-text);
  margin-top: 6px;
  text-align: center;
  z-index: 1;
}
`;

class NodeProgressBar {
  constructor() {
    this.nodeProgressMap = new Map(); // 노드 ID별 진행 상태 저장
    this.nodeTimeMap = new Map(); // 노드별 시작 시간 저장
    this.estimatedTimeMap = new Map(); // 노드별 예상 완료 시간 저장
    this.addStyle();
    this.setupListeners();
    this.setupNodeDrawing();
  }

  addStyle() {
    // CSS 스타일 추가
    const style = document.createElement('style');
    style.textContent = NODE_PROGRESS_STYLE;
    document.head.appendChild(style);
  }

  setupListeners() {
    // API 이벤트 리스너 설정
    api.addEventListener('progress', this.handleProgress.bind(this), false);
    api.addEventListener('execution_start', this.handleExecutionStart.bind(this), false);
    api.addEventListener('executed', this.handleExecuted.bind(this), false);
    api.addEventListener('execution_error', this.handleExecutionError.bind(this), false);
    
    // 노드 추가/제거 리스너
    app.graph.onNodeAdded = this.wrapFunction(app.graph.onNodeAdded, (node) => {
      this.setupNodeUI(node);
    });
  }

  wrapFunction(originalFunction, newFunction) {
    return function() {
      const result = originalFunction ? originalFunction.apply(this, arguments) : undefined;
      newFunction.apply(this, arguments);
      return result;
    };
  }

  setupNodeDrawing() {
    // 노드 그리기 함수 재정의
    const self = this;
    
    // 이미 존재하는 노드에 UI 추가
    for (const node of app.graph.getNodes()) {
      this.setupNodeUI(node);
    }
  }

  setupNodeUI(node) {
    if (!node || node._hasProgressBar) return;
    node._hasProgressBar = true;
    
    // 노드 DOM 요소에 진행 상태 표시줄 컨테이너 추가
    const onDrawBackground = node.onDrawBackground;
    
    node.onDrawBackground = function(ctx) {
      if (onDrawBackground) {
        onDrawBackground.call(this, ctx);
      }
      
      // 노드 ID로 진행 상태 확인
      const progress = NodeProgressBar.instance.nodeProgressMap.get(this.id);
      const estimatedTime = NodeProgressBar.instance.estimatedTimeMap.get(this.id);
      
      // 진행 상태가 있으면 표시
      if (progress !== undefined) {
        const titleHeight = 30; // 노드 제목 영역의 대략적인 높이
        
        // 노드 위젯(구성)의 Y위치 찾기
        const widgetStartY = titleHeight + 5;
        let widgetY = widgetStartY;
        
        if (this.widgets && this.widgets.length > 0) {
          // 첫 번째 위젯 바로 위에 표시
          widgetY = this.widgets[0].last_y ? 
            this.widgets[0].last_y - 16 : // 위젯 위치가 있으면 그 위에 표시
            widgetStartY; // 없으면 기본값 사용
        }
        
        // 진행 상태 표시줄 배경
        ctx.fillStyle = "#333";
        ctx.fillRect(5, widgetY, this.size[0] - 10, 8);
        
        // 진행 상태 표시줄
        ctx.fillStyle = progress >= 100 ? "#4CAF50" : "#2196F3";
        ctx.fillRect(5, widgetY, (progress / 100) * (this.size[0] - 10), 8);
        
        // 진행률 텍스트
        ctx.fillStyle = "#FFF";
        ctx.font = "10px Arial";
        ctx.textAlign = "center";
        
        let statusText = `${progress}%`;
        if (estimatedTime && estimatedTime > 0) {
          const remainingSeconds = Math.max(0, Math.floor((estimatedTime - Date.now()) / 1000));
          if (remainingSeconds > 0) {
            const minutes = Math.floor(remainingSeconds / 60);
            const seconds = remainingSeconds % 60;
            statusText += ` (남은 시간: ${minutes}:${seconds.toString().padStart(2, '0')})`;
          }
        }
        
        // 텍스트 그림자로 가독성 향상
        ctx.shadowColor = "rgba(0,0,0,0.7)";
        ctx.shadowBlur = 2;
        ctx.shadowOffsetX = 1;
        ctx.shadowOffsetY = 1;
        
        ctx.fillText(statusText, this.size[0] / 2, widgetY + 16);
        
        // 그림자 초기화
        ctx.shadowColor = "transparent";
        
        // 노드 크기 조정 (공간 확보)
        if (!this._originalHeight) {
          this._originalHeight = this.size[1];
          this.size[1] += 20; // 진행 상태 표시줄 공간 추가
        }
      } else if (this._originalHeight) {
        // 진행 상태가 없으면 원래 크기로 복원
        this.size[1] = this._originalHeight;
        this._originalHeight = null;
      }
    };
  }

  handleProgress(event) {
    const detail = event.detail;
    if (!detail || !detail.node) return;
    
    const { node: nodeId, value, max } = detail;
    const progress = Math.floor((value / max) * 100);
    
    // 진행 상태 업데이트
    if (!isNaN(progress) && progress >= 0 && progress <= 100) {
      this.nodeProgressMap.set(nodeId, progress);
      
      // 예상 완료 시간 계산
      const startTime = this.nodeTimeMap.get(nodeId);
      if (startTime && progress > 0) {
        const elapsed = Date.now() - startTime;
        const totalEstimated = elapsed / (progress / 100);
        const estimatedCompletion = startTime + totalEstimated;
        this.estimatedTimeMap.set(nodeId, estimatedCompletion);
      }
      
      // 노드 다시 그리기
      const node = app.graph.getNodeById(nodeId);
      if (node) {
        node.setDirtyCanvas(true, true);
      }
    }
  }

  handleExecutionStart(event) {
    // 실행 시작 시 모든 노드의 진행 상태 초기화
    this.nodeProgressMap.clear();
    this.estimatedTimeMap.clear();
    this.nodeTimeMap.clear();
    
    // 현재 실행 중인 모든 노드의 시작 시간 설정
    const startTime = Date.now();
    const activeNodes = app.graph._nodes.filter(n => n.mode === 1); // 활성 노드만 선택
    
    activeNodes.forEach(node => {
      this.nodeTimeMap.set(node.id, startTime);
    });
    
    // 모든 노드 다시 그리기
    for (const node of app.graph.getNodes()) {
      node.setDirtyCanvas(true, true);
    }
  }

  handleExecuted(event) {
    const detail = event.detail;
    if (!detail || !detail.node) return;
    
    const nodeId = detail.node;
    
    // 해당 노드의 진행 상태를 100%로 설정
    this.nodeProgressMap.set(nodeId, 100);
    
    // 실행 완료 시간 기록
    const startTime = this.nodeTimeMap.get(nodeId);
    if (startTime) {
      const executionTime = Date.now() - startTime;
      console.log(`노드 ${nodeId} 실행 완료: ${executionTime}ms 소요`);
    }
    
    // 예상 시간 맵에서 제거
    this.estimatedTimeMap.delete(nodeId);
    
    // 노드 다시 그리기
    const node = app.graph.getNodeById(nodeId);
    if (node) {
      node.setDirtyCanvas(true, true);
    }
  }

  handleExecutionError(event) {
    // 실행 오류 시 처리 (선택적)
  }
}

// 싱글톤 인스턴스 생성
NodeProgressBar.instance = new NodeProgressBar();

// ComfyUI 확장 등록
app.registerExtension({
  name: "Develop.nodeProgressBar",
  setup: () => {
    console.log("노드 진행 상태 표시 기능이 초기화되었습니다.");
  }
}); 