import React from "react";
import * as M from "../styles/StyledMain";
import { useNavigate } from "react-router-dom";

const Main = () => {
  const navigate = useNavigate();

  const goProject = () => navigate(`/project`);
  const goHome = () => navigate(`/`);
  const goAbout = () => navigate(`/about`);
  const goModel = () => navigate(`/model`);

  return (
    <M.Container>
      <M.Blur1 />
      <M.Blur2 />
      <M.Blur3 />
      <M.Blur4 />
      <M.Nav>
        <M.Logo>
          <M.CYC>
            <img
              src={`${process.env.PUBLIC_URL}/images/logo_c.png`}
              alt="logo"
            />
            <div>CYClone</div>
          </M.CYC>
          <M.UNDP>
            <img
              src={`${process.env.PUBLIC_URL}/images/logo_u.png`}
              alt="logo"
            />
            <div>UNDP</div>
          </M.UNDP>
        </M.Logo>
        <M.Comp>
          <M.Ele style={{ cursor: "pointer" }} onClick={goHome}>
            <img
              src={`${process.env.PUBLIC_URL}/images/home_c.svg`}
              alt="home"
            />
            <div style={{ color: "#6BB1FF" }}>Home</div>
          </M.Ele>
          <M.Ele style={{ cursor: "pointer" }} onClick={goProject}>
            <img
              src={`${process.env.PUBLIC_URL}/images/project_w.svg`}
              alt="home"
            />
            <div>Project</div>
          </M.Ele>
          <M.Ele style={{ cursor: "pointer" }} onClick={goModel}>
            <img
              src={`${process.env.PUBLIC_URL}/images/model_w.svg`}
              alt="home"
            />
            <div>Model</div>
          </M.Ele>
          <M.Ele style={{ cursor: "pointer" }} onClick={goAbout}>
            <img
              src={`${process.env.PUBLIC_URL}/images/about_w.svg`}
              alt="home"
            />
            <div>About us</div>
          </M.Ele>
        </M.Comp>
      </M.Nav>
      <M.Title>2025 UNDP Data Dive for Development</M.Title>
      <M.Tit>LifeSim</M.Tit>
      <M.Sub>
        Scenario-Based ODA-Life Expectancy Simulation System for Ethiopia
      </M.Sub>
      <M.Det>
        Data-driven simulations revealing how ODA investments impact life
        expectancy and socio-economic <br />
        development in Ethiopia, enabling scenario-based policy insights for
        sustainable development planning.
      </M.Det>
      <M.Button>
        <M.PBtn style={{ cursor: "pointer" }} onClick={goProject}>
          About Project
        </M.PBtn>
        <M.MBtn style={{ cursor: "pointer" }} onClick={goModel}>
          About Model
        </M.MBtn>
      </M.Button>
      <M.Outcomes>Expected Outcomes</M.Outcomes>
      <M.OutDet>
        Data-driven guidance for scenario-based ODA planning and sustainable
        development
      </M.OutDet>
      <M.Dev>
        <M.Component>
          <M.Img>
            <img
              src={`${process.env.PUBLIC_URL}/images/first.png`}
              id="first"
              alt="first"
            />
          </M.Img>
          <M.Frame>
            <M.BackgroundImage src="/images/frame.png" alt="background" />
            <M.CTit>Life Simulation</M.CTit>
            <M.CDet>
              Scenario-based prediction of life
              <br />
              expectancy under different ODA
              <br />
              interventions
            </M.CDet>
          </M.Frame>
        </M.Component>
        <M.Component>
          <M.Img>
            <img
              src={`${process.env.PUBLIC_URL}/images/second.png`}
              id="second"
              alt="first"
            />
          </M.Img>
          <M.Frame>
            <M.BackgroundImage src="/images/frame.png" alt="background" />
            <M.CTit>Impact Comparison</M.CTit>
            <M.CDet>
              Assessment of short-, medium-,
              <br />
              and long-term effects of sectoral
              <br />
              ODA investments (lag analysis)
            </M.CDet>
          </M.Frame>
        </M.Component>
        <M.Component>
          <M.Img>
            <img
              src={`${process.env.PUBLIC_URL}/images/third.png`}
              id="third"
              alt="first"
            />
          </M.Img>
          <M.Frame>
            <M.BackgroundImage src="/images/frame.png" alt="background" />
            <M.CTit>Optimal Allocation</M.CTit>
            <M.CDet>
              Identification of the best ODA
              <br />
              combinations through sensitivity
              <br />
              analysis for maximizing
              <br />
              development outcomes
            </M.CDet>
          </M.Frame>
        </M.Component>
      </M.Dev>
    </M.Container>
  );
};

export default Main;
