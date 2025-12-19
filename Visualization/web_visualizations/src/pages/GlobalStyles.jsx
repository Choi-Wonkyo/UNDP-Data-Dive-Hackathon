// /mnt/data/GlobalStyles.jsx
import { createGlobalStyle } from "styled-components";

const GlobalStyle = createGlobalStyle`
@font-face {
  font-family: 'Pretendard';
  src: url('/fonts/pretendard/Pretendard-Regular.woff2') format('woff2');
  font-weight: 400;
  font-display: swap;
}

@font-face {
  font-family: 'Pretendard';
  src: url('/fonts/pretendard/Pretendard-Medium.woff2') format('woff2');
  font-weight: 500;
  font-display: swap;
}

@font-face {
  font-family: 'Pretendard';
  src: url('/fonts/pretendard/Pretendard-SemiBold.woff2') format('woff2');
  font-weight: 600;
  font-display: swap;
}

@font-face {
  font-family: 'Pretendard';
  src: url('/fonts/pretendard/Pretendard-Bold.woff2') format('woff2');
  font-weight: 700;
  font-display: swap;
}

  * {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
    font-family: 'Pretendard Variable', sans-serif;
  }

  input::placeholder,
  textarea::placeholder {
    font-family: 'Pretendard Variable', sans-serif;
    color: #999;
    font-weight: 400;
  }
    
  html, body {
  width: 100%;
  margin: 0;
  padding: 0;
  overflow-x: hidden;
  background-color: #000; /* ✅ 검정 배경 적용 */
  }

    
  #root {
  width: 100%;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  }

  ::-webkit-scrollbar {
  width: 10px;
  background-color: #000; /* ✅ 스크롤바 트랙 색 */
}

::-webkit-scrollbar-track {
  background-color: #000; /* ✅ 트랙 색도 검정으로 */
}

::-webkit-scrollbar-thumb {
  background-color: #000;
  border-radius: 8px;
}


`;

export default GlobalStyle;
