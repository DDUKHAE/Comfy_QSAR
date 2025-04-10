/**
 * Progress Bar Extension for ComfyUI
 */
import "../../pre_release/web/js/nodeProgressBar.js"; // 노드 내부 진행 상태 표시 기능만 가져오기

// CSS 로드
const styleElement = document.createElement("link");
styleElement.rel = "stylesheet";
styleElement.type = "text/css";
styleElement.href = "./extensions/develop/progress_bar.css";
document.head.appendChild(styleElement);

// 확장 초기화 함수
export async function init() {
    console.log("노드 진행 상태 표시 기능이 초기화되었습니다.");
    return true;
}

// 자동 초기화
init().catch(err => {
    console.error(`진행 상태 표시 기능 초기화 실패:`, err);
}); 