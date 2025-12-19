import { styled } from "styled-components";

export const Container = styled.div`
  position: relative;
  margin: 0 auto;
  margin-top: 0px;
  min-height: 100dvh;
  padding: 0; /* 불필요한 패딩 제거 */
  box-sizing: border-box; /* 패딩이 width에 포함되도록 설정 */
  display: flex;
  flex-direction: column;
  width: 100%;
  flex-shrink: 0;
  align-items: center;
  background: linear-gradient(180deg, #000 0%, #032f61 50.48%, #000 100%);
  padding-bottom: 335px;
`;

export const Back = styled.div`
  z-index: 0;
  position: absolute;
  width: 100%;

  img {
    margin-top: 80px;
  }
`;

export const Nav = styled.div`
  display: flex;
  width: 100%;
  height: 80px;
  top: 0;
  left: 0;
  padding: 28px 140px 27px 140px;
  justify-content: space-between;
  align-items: center;
  background: rgba(255, 255, 255, 0.1);
  z-index: 10;
  flex-direction: row;
  position: fixed;
  backdrop-filter: blur(10px); /* 선택 사항: 반투명 배경에 흐림 효과 */
  box-sizing: border-box; /* 패딩 포함 */
`;

export const Logo = styled.div`
  display: flex;
  align-items: center;
  gap: 60px;
`;

export const CYC = styled.div`
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 10px;

  img {
    width: 25px;
    height: 25px;
  }

  div {
    color: #fff;
    font-family: Pretendard;
    font-size: 15px;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
    letter-spacing: 1.5px;
  }
`;

export const UNDP = styled.div`
  display: flex;
  flex-direction: row;
  gap: 10px;
  align-items: center;

  img {
    width: 25px;
    height: 25px;
  }

  div {
    color: #fff;
    font-family: Pretendard;
    font-size: 15px;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
    letter-spacing: 1.5px;
  }
`;

export const Comp = styled.div`
  display: flex;
  align-items: center;
  gap: 50px;
`;

export const Ele = styled.div`
  display: flex;
  flex-direction: row;
  gap: 10px;
  align-items: center;

  img {
    width: 18px;
    height: 18px;
    aspect-ratio: 1/1;
  }

  div {
    color: #fff;
    font-family: Pretendard;
    font-size: 15px;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  }
`;

export const Title = styled.div`
  font-family: Pretendard;
  font-size: 80px;
  font-style: normal;
  font-weight: 700;
  margin-top: 208px;
  line-height: normal;
  background: linear-gradient(
    90deg,
    rgba(149, 255, 160, 0.8) 0%,
    rgba(132, 180, 215, 0.9) 48%,
    #0063ac 100%
  );
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  z-index: 3;
`;

export const Det_70 = styled.div`
  color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: 35px; /* 140% */
  margin-top: 70px;
  z-index: 3;
`;

export const Det_40 = styled.div`
  color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: 35px; /* 140% */
  margin-top: 40px;
  z-index: 3;
`;

export const Semi = styled.div`
  color: #fff;
  font-family: Pretendard;
  font-size: 60px;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  letter-spacing: 3px;
  margin-top: 200px;
  z-index: 3;
`;

export const Det_99 = styled.div`
  color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: 35px; /* 140% */
  margin-top: 99px;
  z-index: 3;
`;

export const List = styled.div`
  width: 100%;
  display: flex;
  flex-direction: row;
  gap: 20px;
  margin-top: 40px;
  align-items: center;
  justify-content: center;
  z-index: 3;
`;

export const Box = styled.div`
  display: flex;
  width: 307px;
  height: 122px;
  padding: 26px 17px;
  justify-content: center;
  align-items: center;
  border-radius: 20px;
  background: #213370;
  box-shadow: 2px 3px 4px 0 rgba(255, 255, 255, 0.25) inset,
    3px 3px 4px 0 rgba(255, 255, 255, 0.1);
  color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: 35px; /* 140% */
  z-index: 3;
`;

export const B1 = styled.div`
  color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: 35px; /* 140% */
  margin-top: 110px;
  z-index: 3;
`;

export const B2 = styled.div`
  color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: 35px; /* 140% */
  margin-top: 50px;
  z-index: 3;
  margin-bottom: 44px;
`;

export const Profile = styled.div`
  width: 100%;
  display: flex;
  flex-direction: row;
  gap: 60px;
  align-items: center;
  justify-content: center;
  margin-top: 100px;
`;

export const Card = styled.div`
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  transition: all 0.35s ease;
  cursor: pointer;

  /* 선택된 Card → 커짐 */
  ${({ active }) =>
    active &&
    `
        transform: scale(1.33); /* 372 → 495 정도로 커짐 */
  `}

  /* 선택되지 않은 Card → 작아짐 */
  ${({ shrink }) =>
    shrink &&
    `
        transform: scale(0.85);
        opacity: 0.8;
  `}
`;

export const Prof = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 20px;
  transition: all 0.35s ease;

  img {
    width: 372px;
    height: 395px;
    border-radius: 30px;
    object-fit: cover;
    transition: all 0.35s ease;

    /* 선택된 카드 → 이미지 확대 */
    ${({ active }) =>
      active &&
      `
        transform: scale(1.33); /* 372 → 495 정도로 커짐 */
      `}

    /* 선택되지 않은 카드 → 이미지 축소 */
    ${({ shrink }) =>
      shrink &&
      `
        transform: scale(0.85);
        opacity: 0.8;
      `}
  }

  div {
    color: #fff;
    text-align: center;
    font-family: Pretendard;
    font-size: 30px;
    font-style: normal;
    font-weight: 600;
    letter-spacing: 1.5px;
    z-index: 5;
  }
`;

export const Overlay = styled.div`
  position: absolute;
  top: 0;
  width: 372px; /* 이미지 기본 width */
  height: 395px; /* 이미지 기본 height */
  border-radius: 30px;
  background: rgba(0, 0, 0, 0.6);
  pointer-events: none;

  transform: scale(${({ active }) => (active ? 1 : 1)});
  transform-origin: center;

  opacity: ${({ active }) => (active ? 1 : 0)};
  transition: opacity 0.35s ease, transform 0.35s ease;
`;

export const DetailText = styled.div`
  position: absolute;
  width: 460px;
  display: flex;
  height: 525px;
  flex-direction: column;
  align-items: start;
  // justify-content: center;
  z-index: 20;
  padding: 0;
  margin-top: -30px;
  margin-left: 11px;

  opacity: ${({ active }) => (active ? 1 : 0)};
  transform: ${({ active }) =>
    active
      ? "translateY(0) scale(0.75)" // 📌 Card가 1.33배니까 1/1.33로 보정
      : "translateY(15px) scale(1)"};
  transform-origin: center;
  transition: all 0.35s ease;

  #intro {
    color: #fff;
    font-family: Pretendard;
    font-size: 24px;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
    letter-spacing: 1.375px;
    text-align: left;
  }

  #body {
    color: #fff;
    font-family: Pretendard;
    font-size: 18px;
    font-style: normal;
    font-weight: 500;
    line-height: 28.6px; /* 130% */
    letter-spacing: 1.1px;
    margin-top: 20px;
    text-align: left;
  }

  #footer {
    color: #fff;
    font-family: Pretendard;
    font-size: 18px;
    font-style: normal;
    font-weight: 500;
    line-height: 28px; /* 127.273% */
    letter-spacing: 0.44px;
    margin-top: 30px;
    text-align: left;
  }
`;

export const Blur1 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 2064.508px;
  height: 1792.637px;
  transform: rotate(-178.332deg);
  z-index: 0;
  pointer-events: none;
  position: absolute;
  top: 231px;
  right: 0;
  border-radius: 2064.508px;
  background: conic-gradient(
    from 180deg at 50% 50%,
    rgba(0, 72, 153, 0.4) 107.30769395828247deg,
    rgba(149, 185, 255, 0.8) 233.6249542236328deg,
    rgba(215, 149, 255, 0.8) 310.4999613761902deg
  );
  filter: blur(145.06704711914062px);
  opacity: 0.6; /* 투명도 더 높게 */
`;

export const Blur2 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 1454.479px;
  height: 1947.921px;
  transform: rotate(-66.585deg);
  z-index: 0;
  pointer-events: none;
  position: absolute;
  top: 382px;
  left: 10%;
  border-radius: 1947.921px;
  background: conic-gradient(
    from 180deg at 50% 50%,
    rgba(0, 72, 153, 0.3) 0deg,
    #004899 70.49995422363281deg,
    #38608c 162.3749577999115deg,
    #95b9ff 233.6249542236328deg,
    rgba(186, 149, 255, 0.8) 310.4999613761902deg
  );
  mix-blend-mode: overlay;
  filter: blur(119.81165313720703px);
  opacity: 0.6; /* 투명도 더 높게 */
`;
