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
  max-width: 100%;
  align-items: center;
  background: #000;
  padding-bottom: 400px;
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
  z-index: 5;
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
  color: #fff;
  font-family: Pretendard;
  font-size: 20px;
  font-style: normal;
  font-weight: 500;
  line-height: normal;
  margin-top: 212px;
  z-index: 2;
`;

export const Tit = styled.div`
  font-family: Pretendard;
  font-size: 80px;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  letter-spacing: 4px;
  background: linear-gradient(
    90deg,
    rgba(149, 255, 160, 0.8) 0%,
    #84b4d7 48.08%,
    #0063ac 100%
  );
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-top: 60px;
  z-index: 2;
`;

export const Sub = styled.div`
  color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: normal;
  margin-top: 80px;
  z-index: 2;
`;

export const Det = styled.div`
  color: #fff;
  font-family: Pretendard;
  font-size: 20px;
  font-style: normal;
  font-weight: 500;
  line-height: 35px; /* 175% */
  margin-top: 50px;
  z-index: 2;
`;

export const Button = styled.div`
  display: flex;
  width: 630px;
  justify-content: space-between;
  align-items: center;
  margin-top: 100px;
  z-index: 2;
`;

export const PBtn = styled.div`
  display: flex;
  width: 200px;
  height: 70px;
  justify-content: center;
  align-items: center;
  flex-shrink: 0;
  border-radius: 50px;
  border: 3px solid #fff;
  color: #fff;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: normal;
  z-index: 2;

  &:hover {
    box-shadow: 0 0 15px 6px rgba(255, 255, 255, 0.6);
    transform: scale(1.05);
    backdrop-filter: blur(4px);
  }
`;

export const MBtn = styled.div`
  display: flex;
  width: 200px;
  height: 70px;
  justify-content: center;
  align-items: center;
  flex-shrink: 0;
  border-radius: 50px;
  border: 3px solid #004899;
  background: #004899;
  color: #fff;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: normal;
  z-index: 2;
  transition: all 0.3s ease;
  position: relative;

  &:hover {
    box-shadow: 0 0 20px 8px rgba(0, 72, 153, 0.6);
    transform: scale(1.05);
    backdrop-filter: blur(4px);
  }
`;

export const Outcomes = styled.div`
  color: #fff;
  font-family: Pretendard;
  font-size: 60px;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  letter-spacing: 3px;
  margin-top: 180px;
  z-index: 2;
`;

export const OutDet = styled.div`
  // color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: normal;
  margin-top: 33.33px;
  z-index: 2;
  background: linear-gradient(90deg, #c2c2c2 0%, #757575 100%);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;

export const Dev = styled.div`
  margin-top: 204px;
  gap: 30px;
  display: flex;
  flex-direction: row;
  align-items: center;
  z-index: 2;
`;

export const Component = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  z-index: 2;
`;

export const Img = styled.div`
  width: 200px;
  height: 125px;
  // background: #d9d9d9;
  position: absolute;
  margin-top: -44px;
  z-index: 3;
  align-items: center;
  justify-content: center;
  display: flex;

  #first {
    width: 111px;
    height: 111px;
    aspect-ratio: 1/1;
  }

  #second {
    width: 113px;
    height: 133px;
    aspect-ratio: 113/133;
  }

  #third {
    width: 146px;
    height: 133px;
    aspect-ratio: 146/133;
  }
`;

export const Frame = styled.div`
  width: 440px;
  height: 450px;
  display: flex;
  flex-direction: column;
  text-align: center;
  align-items: center;
  gap: 60px;
  position: relative;
  overflow: hidden;
  z-index: 2;
`;

export const BackgroundImage = styled.img`
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  z-index: -1;

  opacity: 0.22; /* fill 비슷한 효과 */
  filter: drop-shadow(1px 1px 1px rgba(196, 196, 196, 0.8))
    drop-shadow(-1px -1px 1px rgba(196, 196, 196, 0.8));
`;

export const CTit = styled.div`
  color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 30px;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  margin-top: 151px;
  z-index: 2;
`;

export const CDet = styled.div`
  color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: 35px; /* 140% */
  z-index: 2;
`;

export const Blur1 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 1258.453px;
  height: 1223.605px;
  transform: rotate(175.68deg);
  z-index: 0;
  pointer-events: none;
  position: absolute;
  top: 38.93px;
  left: -549.78px;
  border-radius: 1258.453px;
  background: conic-gradient(
    from 180deg at 50% 50%,
    #004899 107.30769395828247deg,
    #95b9ff 233.6249542236328deg,
    #d795ff 310.4999613761902deg
  );
  filter: blur(145.06704711914062px);
`;

export const Blur2 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 1233.78px;
  height: 1533.585px;
  transform: rotate(-137.443deg);
  z-index: 0;
  pointer-events: none;
  position: absolute;
  top: -332.08px;
  left: -927.26px;
  border-radius: 1533.585px;
  background: conic-gradient(
    from 180deg at 50% 50%,
    #95b9ff 233.6249542236328deg,
    #d795ff 310.4999613761902deg
  );
  mix-blend-mode: multiply;
  filter: blur(119.81165313720703px);
`;

export const Blur3 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 1071.631px;
  height: 1254.474px;
  transform: rotate(-56.448deg);
  z-index: 0;
  pointer-events: none;
  position: absolute;
  top: 51.49px;
  left: -1041px;
  border-radius: 1254.474px;
  background: conic-gradient(
    from 180deg at 50% 50%,
    #004899 0deg,
    #004899 70.49995422363281deg,
    #38608c 162.3749577999115deg,
    #95b9ff 233.6249542236328deg,
    #ba95ff 310.4999613761902deg
  );
  mix-blend-mode: overlay;
  filter: blur(119.81165313720703px);
`;

export const Blur4 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 944.896px;
  height: 1313.361px;
  z-index: 0;
  pointer-events: none;
  position: absolute;
  top: 861.52px;
  right: -466.07px;
  border-radius: 1313.361px;
  background: conic-gradient(
    from 2deg at 50% 50%,
    rgba(0, 72, 153, 0.8) 70.49995422363281deg,
    rgba(149, 185, 255, 0.8) 192.11538791656494deg,
    rgba(215, 149, 255, 0.8) 310.4999613761902deg
  );
  filter: blur(250px);
`;
