import { styled } from "styled-components";

export const Container = styled.div`
  position: relative;
  margin: 0 auto;
  margin-top: 0px;
  min-height: 100dvh;
  padding-bottom: 208px;
  box-sizing: border-box; /* 패딩이 width에 포함되도록 설정 */
  display: flex;
  flex-direction: column;
  width: 100%;
  flex-shrink: 0;
  align-items: center;
  background: linear-gradient(
    180deg,
    #000 0%,
    rgba(0, 72, 153, 0.2) 51.92%,
    #000 100%
  );
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
  z-index: 2;
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
  margin-top: 230px;
`;

export const Sub = styled.div`
  margin-top: 120px;
  color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: 35px; /* 140% */
`;

export const Tit = styled.div`
  margin-top: 200px;
  color: #fff;
  font-family: Pretendard;
  font-size: 60px;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  letter-spacing: 3px;
  margin-bottom: 40px;
`;

export const Det = styled.div`
  margin-top: 60px;
  color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: 35px; /* 140% */
`;

export const List = styled.div`
  margin-top: 190px;
  gap: 60px;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

export const Box = styled.div`
  width: 1187px;
  height: 558px;
  border-radius: 80px;
  background: #0e0909;
  align-items: center;
  display: flex;
  flex-direction: column;
  z-index: 1;
  position: relative;
  overflow: hidden;
`;

export const Img1 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 336.794px;
  height: 327.624px;
  transform: rotate(176.521deg);
  position: absolute;
  top: -56px;
  left: 882px;
  z-index: 0;
  pointer-events: none;
  background: linear-gradient(
    180deg,
    rgba(24, 75, 255, 0) 0%,
    rgba(0, 194, 255, 0.7) 100%
  );
  filter: blur(50px);
`;

export const Img3 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 336.794px;
  height: 327.624px;
  transform: rotate(176.521deg);
  position: absolute;
  top: 49.17px;
  left: 1018.69px;
  background: linear-gradient(
    180deg,
    rgba(24, 75, 255, 0) 0%,
    rgba(23, 74, 255, 0.7) 100%
  );
  filter: blur(50px);
  z-index: 0;
  pointer-events: none;
`;

export const Img2 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 404px;
  height: 393px;
  position: absolute;
  top: 344px;
  left: -66px;
  z-index: 0;
  pointer-events: none;
  background: linear-gradient(
    180deg,
    rgba(24, 75, 255, 0) 0%,
    rgba(0, 194, 255, 0.7) 100%
  );
  filter: blur(50px);
`;

export const Img4 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 404px;
  height: 393px;
  position: absolute;
  z-index: 0;
  pointer-events: none;
  top: 208px;
  left: -222px;
  background: linear-gradient(
    180deg,
    rgba(24, 75, 255, 0) 0%,
    rgba(23, 74, 255, 0.7) 100%
  );
  filter: blur(50px);
`;

export const BTit = styled.div`
  color: #fff;
  font-family: Pretendard;
  font-size: 60px;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  letter-spacing: 3px;
  margin-top: 100px;
`;

export const BDet = styled.div`
  display: flex;
  flex-direction: column;
  gap: 20px;
  align-items: center;
  text-align: center;
  margin-top: 111px;
  z-index: 3;

  div {
    color: #fff;
    text-align: center;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: 35px; /* 175% */
  }
`;

export const Blur1 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 182.932px;
  height: 302.918px;
  transform: rotate(-116.811deg);
  position: absolute;
  z-index: 0;
  pointer-events: none;
  top: 371.38px;
  left: 1020.44px;
  background: linear-gradient(
    180deg,
    rgba(24, 75, 255, 0) 0%,
    rgba(0, 194, 255, 0.7) 100%
  );
  filter: blur(50px);
`;

export const Blur2 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 285.183px;
  height: 299.448px;
  transform: rotate(28.564deg);
  position: absolute;
  z-index: 0;
  pointer-events: none;
  top: 75.48px;
  left: -213.34px;
  background: linear-gradient(
    180deg,
    rgba(24, 75, 255, 0) 0%,
    rgba(0, 194, 255, 0.7) 100%
  );
  filter: blur(50px);
`;

export const Blur3 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 285.183px;
  height: 299.448px;
  transform: rotate(28.564deg);
  position: absolute;
  z-index: 0;
  pointer-events: none;
  top: -69.19px;
  left: -260.51px;
  background: linear-gradient(
    180deg,
    rgba(24, 75, 255, 0) 0%,
    rgba(23, 74, 255, 0.7) 100%
  );
  filter: blur(50px);
`;

export const Blur4 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 182.932px;
  height: 302.918px;
  transform: rotate(-116.811deg);
  position: absolute;
  z-index: 0;
  pointer-events: none;
  top: 481.7px;
  left: 957.75px;
  background: linear-gradient(
    180deg,
    rgba(24, 75, 255, 0) 0%,
    rgba(23, 74, 255, 0.7) 100%
  );
  filter: blur(50px);
`;

export const G1 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 292.183px;
  height: 306.897px;
  transform: rotate(107.533deg);
  position: absolute;
  z-index: 0;
  pointer-events: none;
  top: -106.42px;
  left: -111px;
  background: linear-gradient(
    180deg,
    rgba(24, 75, 255, 0) 0%,
    rgba(0, 194, 255, 0.7) 100%
  );
  filter: blur(50px);
`;

export const G2 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 292.183px;
  height: 306.897px;
  transform: rotate(107.533deg);
  position: absolute;
  z-index: 0;
  pointer-events: none;
  top: -182.01px;
  left: 24.26px;
  background: linear-gradient(
    180deg,
    rgba(24, 75, 255, 0) 0%,
    rgba(23, 74, 255, 0.7) 100%
  );
  filter: blur(50px);
`;

export const G3 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 415.663px;
  height: 888.564px;
  transform: rotate(96.937deg);
  position: absolute;
  z-index: 0;
  pointer-events: none;
  left: 1230px;
  background: linear-gradient(
    180deg,
    rgba(24, 75, 255, 0) 0%,
    rgba(0, 194, 255, 0.7) 100%
  );
  filter: blur(50px);
`;

export const G4 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 415.663px;
  height: 488.891px;
  transform: rotate(96.937deg);
  position: absolute;
  z-index: 0;
  pointer-events: none;
  top: 56.92px;
  left: 1130.88px;
  background: linear-gradient(
    180deg,
    rgba(24, 75, 255, 0) 0%,
    rgba(23, 74, 255, 0.5) 100%
  );
  filter: blur(50px);
`;

export const Conclusion = styled.div`
  color: #fff;
  font-family: Pretendard;
  font-size: 60px;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  letter-spacing: 3px;
  margin-top: 200px;
`;

export const Con = styled.div`
  margin-top: 100px;
  gap: 60px;
  display: flex;
  flex-direction: column;

  div {
    color: #fff;
    text-align: center;
    font-family: Pretendard;
    font-size: 25px;
    font-style: normal;
    font-weight: 500;
    line-height: 35px; /* 140% */
  }
`;

export const Github = styled.div`
  margin-top: 250px;
  width: 430px;
  height: 50px;
  border-radius: 50px;
  background: #004899;
  display: flex;
  flex-direction: row;
  gap: 10px;
  align-items: center;
  justify-content: center;

  img {
    width: 40px;
    height: 40px;
  }

  &:hover {
    box-shadow: 0 0 20px 10px rgba(0, 72, 153, 0.5);
    backdrop-filter: blur(4px); /* 약간의 블러 효과 */
    transform: scale(1.03); /* 약간 커지는 효과 */
  }
`;

export const GDet = styled.div`
  display: flex;
  flex-direction: row;
  align-items: center;

  #text {
    color: #fff;
    text-align: center;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
    letter-spacing: 0.6px;
  }

  #bold {
    margin-left: 5px;
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
    letter-spacing: 0.6px;
  }
`;
