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
  background: linear-gradient(180deg, #000 41.83%, #011730 56.96%, #000 100%);
  padding-bottom: 400px;
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
  z-index: 1;
`;

export const Flow1 = styled.div`
  display: flex;
  flex-direction: row;
  gap: 32px;
  margin-top: 80px;
  z-index: 1;
  align-items: center;
  justify-content: center;
  width: 100%;

  img {
    width: 45px;
    height: 26.848px;
  }
`;

export const Flow2 = styled.div`
  display: flex;
  flex-direction: row;
  gap: 32px;
  margin-top: 50px;
  z-index: 1;
  align-items: center;
  justify-content: center;
  width: 100%;

  img {
    width: 45px;
    height: 26.848px;
  }
`;

export const Set = styled.div`
  display: flex;
  flex-direction: column;
  gap: -11px;
  align-items: center;
  justify-content: center;
  z-index: 1;
`;

export const FTit = styled.div`
  width: 290px;
  height: 66px;
  border-radius: 10px 10px 0 0;
  border: 2px solid #4169e1;
  background: #b8d9ff;
  color: #000;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: normal;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 2;
`;

export const FDet = styled.div`
  margin-top: -11px;
  width: 290px;
  height: 142px;
  border-radius: 0 0 10px 10px;
  border: 2px solid #4169e1;
  background: #fff;
  color: #000;
  text-align: center;
  font-family: Pretendard;
  font-size: 20px;
  font-style: normal;
  font-weight: 500;
  line-height: 30px; /* 150% */
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1;
`;

export const Over = styled.div`
  color: #fff;
  font-family: Pretendard;
  font-size: 60px;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  letter-spacing: 3px;
  margin-top: 220px;
  z-index: 1;
`;

export const Goal = styled.div`
  color: #8cc2ff;
  text-align: center;
  font-family: Pretendard;
  font-size: 30px;
  font-style: normal;
  font-weight: 700;
  line-height: 35px; /* 140% */
  margin-top: 120px;
  z-index: 1;
`;

export const GDet = styled.div`
  color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 20px;
  font-style: normal;
  font-weight: 500;
  line-height: 35px; /* 175% */
  margin-top: 32px;
  z-index: 1;
`;

export const User = styled.div`
  margin-top: 100px;
  color: #8cc2ff;
  text-align: center;
  font-family: Pretendard;
  font-size: 30px;
  font-style: normal;
  font-weight: 700;
  line-height: 35px; /* 140% */
  z-index: 1;
`;

export const UDet = styled.div`
  color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 20px;
  font-style: normal;
  font-weight: 500;
  line-height: 35px; /* 175% */
  margin-top: 30px;
  z-index: 1;
`;

export const Descr = styled.div`
  color: #fff;
  font-family: Pretendard;
  font-size: 60px;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  letter-spacing: 3px;
  margin-top: 250px;
  margin-bottom: 20px;
  z-index: 1;
`;

export const Tail = styled.div`
  margin-top: 80px;
  color: #8cc2ff;
  text-align: center;
  font-family: Pretendard;
  font-size: 30px;
  font-style: normal;
  font-weight: 700;
  line-height: 35px; /* 140% */
  z-index: 1;
`;

export const Key = styled.div`
  display: flex;
  width: 1150px;
  height: 300px;
  padding: 36px 0 36px 20px;
  align-items: center;
  justify-content: center;
  border-radius: 25px;
  background: rgba(255, 255, 255, 0.1);
  margin-top: 50px;
  flex-direction: column;
  z-index: 1;

  #title {
    color: #fff;
    text-align: center;
    font-family: Pretendard;
    font-size: 28px;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
    margin-bottom: 24px;
  }

  #detail {
    width: 100%;
    text-align: left;
    margin-left: 52px;
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: 45px;
    letter-spacing: 0.4px;
  }
`;

export const Model = styled.div`
  margin-top: 40px;
  display: flex;
  width: 1150px;
  height: 410px;
  padding: 36px 20px;
  justify-content: center;
  flex-direction: column;
  align-items: center;
  border-radius: 25px;
  background: rgba(255, 255, 255, 0.1);
  z-index: 1;

  #title {
    color: #fff;
    text-align: center;
    font-family: Pretendard;
    font-size: 28px;
    font-style: normal;
    font-weight: 700;
    line-height: normal;
    margin-bottom: 24px;
  }

  #detail {
    width: 100%;
    text-align: left;
    margin-left: 52px;
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: 40px;
    letter-spacing: 0.4px;
    gap: 25px;
    display: flex;
    flex-direction: column;
  }
`;

export const TBox = styled.div`
  margin-top: 50px;
  display: flex;
  width: 1150px;
  height: 325px;
  padding-left: 48px;
  flex-direction: column;
  align-items: start;
  border-radius: 25px;
  background: rgba(255, 255, 255, 0.1);
  z-index: 1;
  justify-content: center;

  #detail {
    display: flex;
    text-align: left;
    flex-direction: column;
    gap: 20px;
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: 35px;
    letter-spacing: 0.4px;
  }
`;

export const Dash = styled.div`
  margin-top: 220px;
  color: #fff;
  font-family: Pretendard;
  font-size: 60px;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  letter-spacing: 3px;
  margin-bottom: 30px;
  z-index: 1;
`;

export const T1 = styled.div`
  margin-top: 70px;
  color: #8cc2ff;
  text-align: center;
  font-family: Pretendard;
  font-size: 30px;
  font-style: normal;
  font-weight: 700;
  line-height: 35px; /* 140% */
  z-index: 1;
`;

export const B1 = styled.div`
  border-radius: 25px;
  background: rgba(255, 255, 255, 0.1);
  margin-top: 30px;
  display: flex;
  width: 1150px;
  height: 285px;
  padding-left: 36px;
  justify-content: center;
  align-items: start;
  flex-direction: column;
  z-index: 1;

  #title {
    color: #fff;
    text-align: center;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: normal;
  }

  #detail {
    margin-left: 27px;
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: 40px;
    letter-spacing: 0.4px;
    margin-top: 21px;
  }

  #ex1 {
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: 35px; /* 175% */
    letter-spacing: 0.4px;
  }
`;

export const B2 = styled.div`
  width: 1150px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  height: 340px;
  padding-left: 20px;
  align-items: start;
  border-radius: 25px;
  background: rgba(255, 255, 255, 0.1);
  margin-top: 30px;
  z-index: 1;

  #detail {
    margin-left: 28px;
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: 40px;
    letter-spacing: 0.4px;
  }

  div {
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: 35px; /* 175% */
    letter-spacing: 0.4px;
  }
`;

export const B3 = styled.div`
  display: flex;
  width: 1150px;
  height: 225px;
  padding-left: 20px;
  border-radius: 25px;
  background: rgba(255, 255, 255, 0.1);
  flex-direction: column;
  margin-top: 30px;
  z-index: 1;
  justify-content: center;
  align-items: start;

  ul {
    margin-left: 28px;
    text-align: left;
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: 40px;
    letter-spacing: 0.4px;
  }

  div {
    margin-top: 30px;
    width: 100%;
    text-align: center;
    margin-right: auto;
    margin-left: -20px;
    color: #fff;
    font-family: Inter;
    font-size: 22px;
    font-style: italic;
    font-weight: 500;
    line-height: 40px; /* 200% */
    letter-spacing: 0.44px;
  }
`;

export const Example = styled.div`
  color: #fff;
  font-family: Pretendard;
  font-size: 60px;
  font-style: normal;
  font-weight: 700;
  z-index: 1;
  line-height: normal;
  letter-spacing: 3px;
  margin-top: 200px;
  margin-bottom: 30px;
`;

export const B4 = styled.div`
  display: flex;
  width: 1150px;
  height: 245px;
  padding-left: 16px;
  align-items: start;
  justify-content: center;
  z-index: 1;
  border-radius: 25px;
  background: rgba(255, 255, 255, 0.1);
  margin-top: 30px;
  flex-direction: column;
  margin-bottom: 5px;

  ul {
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: 40px;
    letter-spacing: 0.4px;
    margin-left: 32px;
  }
`;

export const B5 = styled.div`
  margin-top: 30px;
  display: flex;
  width: 1150px;
  height: 292px;
  padding-left: 16px;
  align-items: start;
  justify-content: center;
  flex-direction: column;
  border-radius: 25px;
  z-index: 1;
  background: rgba(255, 255, 255, 0.1);

  ul {
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: 40px;
    letter-spacing: 0.4px;
    width: 1072px;
    flex-shrink: 0;
    margin-left: 32px;
  }
`;

export const B6 = styled.div`
  margin-top: 30px;
  display: flex;
  width: 1150px;
  z-index: 1;
  height: 245px;
  padding: 0 45px;
  justify-content: center;
  align-items: center;
  border-radius: 25px;
  background: rgba(255, 255, 255, 0.1);
  flex-direction: column;

  div {
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: 40px;
    letter-spacing: 0.4px;
  }
`;

export const Limit = styled.div`
  margin-top: 200px;
  margin-bottom: 30px;
  color: #fff;
  font-family: Pretendard;
  font-size: 60px;
  font-style: normal;
  font-weight: 700;
  line-height: normal;
  z-index: 1;
  letter-spacing: 3px;
`;

export const B7 = styled.div`
  display: flex;
  width: 1150px;
  height: 245px;
  padding-left: 16px;
  align-items: center;
  border-radius: 25px;
  background: rgba(255, 255, 255, 0.1);
  z-index: 1;
  flex-direction: column;
  justify-content: center;
  align-items: start;
  flex-direction: column;
  margin-top: 30px;
  margin-bottom: 5px;

  ul {
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: 40px;
    letter-spacing: 0.4px;
    width: 1104px;
    margin-left: 32px;
    display: flex;
    flex-direction: column;
    gap: 25px;
    text-align: left;
  }
`;

export const B8 = styled.div`
  margin-top: 30px;
  margin-bottom: 5px;
  flex-direction: column;
  display: flex;
  width: 1150px;
  height: 292px;
  z-index: 1;
  padding-left: 16px;
  justify-content: center;
  align-items: start;
  border-radius: 25px;
  background: rgba(255, 255, 255, 0.1);

  ul {
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 500;
    line-height: 40px;
    letter-spacing: 0.4px;
    width: 1072px;
    flex-shrink: 0;
    margin-left: 32px;
    display: flex;
    flex-direction: column;
    gap: 15px;
  }
`;

export const B9 = styled.div`
  margin-top: 30px;
  flex-direction: column;
  display: flex;
  width: 1150px;
  height: 220px;
  padding-left: 16px;
  justify-content: center;
  z-index: 1;
  align-items: start;
  border-radius: 25px;
  background: rgba(255, 255, 255, 0.1);

  ul {
    color: #fff;
    font-family: Pretendard;
    font-size: 20px;
    font-style: normal;
    font-weight: 700;
    line-height: 40px; /* 200% */
    letter-spacing: 0.4px;
    width: 1060px;
    flex-shrink: 0;
    margin-left: 32px;
  }
`;

export const Blur1 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 1157.85px;
  height: 1640.351px;
  z-index: 0;
  pointer-events: none;
  position: absolute;
  top: -529.19px;
  left: -801px;
  border-radius: 1640.351px;
  background: conic-gradient(
    from 2deg at 50% 50%,
    rgba(0, 72, 153, 0.7) 70.49995422363281deg,
    rgba(149, 185, 255, 0.7) 192.11538791656494deg,
    rgba(71, 112, 141, 0.7) 310.4999613761902deg
  );
  filter: blur(250px);
`;

export const Blur2 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 1508.275px;
  height: 1898.938px;
  transform: rotate(15.581deg);
  z-index: 0;
  pointer-events: none;
  position: absolute;
  top: 855.82px;
  right: -1288px;
  border-radius: 1640.351px;
  background: conic-gradient(
    from 194deg at 47.53% 55.37%,
    rgba(0, 72, 153, 0.7) 66.005499958992deg,
    rgba(53, 208, 255, 0.7) 144.62188839912415deg,
    rgba(114, 193, 255, 0.7) 204.23077583312988deg,
    rgba(149, 185, 255, 0.7) 318.87834548950195deg
  );
  filter: blur(300px);
`;

export const Blur3 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 1348.478px;
  height: 1667.03px;
  transform: rotate(0.823deg);
  z-index: 0;
  pointer-events: none;
  position: absolute;
  top: 3737px;
  left: -600px;
  border-radius: 1667.03px;
  background: conic-gradient(
    from 2deg at 50% 50%,
    rgba(0, 72, 153, 0.7) 70.49995422363281deg,
    rgba(149, 185, 255, 0.7) 192.11538791656494deg,
    rgba(215, 149, 255, 0.7) 310.4999613761902deg
  );
  filter: blur(250px);
`;

export const Blur4 = styled.div.attrs({
  "aria-hidden": "true",
  role: "presentation",
})`
  width: 735.214px;
  height: 1370.731px;
  z-index: 0;
  pointer-events: none;
  position: absolute;
  top: 4472px;
  left: -212px;
  border-radius: 1370.731px;
  background: conic-gradient(
    from 2deg at 50% 50%,
    rgba(0, 72, 153, 0.7) 70.49995422363281deg,
    rgba(149, 185, 255, 0.7) 192.11538791656494deg,
    rgba(215, 149, 255, 0.7) 310.4999613761902deg
  );
  filter: blur(250px);
`;

export const Highlight = styled.span`
  background-color: rgba(190, 244, 255, 0.4);
  padding: 2px;
`;

export const Button = styled.div`
  color: #fff;
  text-align: center;
  font-family: Pretendard;
  font-size: 25px;
  font-style: normal;
  font-weight: 500;
  line-height: normal;
  letter-spacing: 0.75px;
  display: flex;
  width: 400px;
  height: 60px;
  padding: 10px;
  justify-content: center;
  align-items: center;
  gap: 10px;
  border-radius: 50px;
  background: #004899;
  margin-top: 200px;

  &:hover {
    box-shadow: 0 0 20px 10px rgba(0, 72, 153, 0.5);
    backdrop-filter: blur(4px); /* 약간의 블러 효과 */
    transform: scale(1.03); /* 약간 커지는 효과 */
  }
`;
