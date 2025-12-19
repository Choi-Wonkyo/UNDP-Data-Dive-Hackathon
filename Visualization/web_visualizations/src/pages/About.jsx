import React, { useState } from "react";
import * as A from "../styles/StyledAbout";
import { Highlight } from "../styles/StyledModel";
import { useNavigate } from "react-router-dom";

const About = () => {
  const navigate = useNavigate();

  const goProject = () => navigate(`/project`);
  const goHome = () => navigate(`/`);
  const goAbout = () => navigate(`/about`);
  const goModel = () => navigate(`/model`);

  const [activeIndex, setActiveIndex] = useState(null);

  const profiles = [
    {
      img: "/images/new_wonkyo.png",
      name: "Choi Wonkyo",
      desc: {
        intro: "Team Leader & Visualization Lead",
        body: "Department of Data Science,\nDongduk Women's University (Expected Graduation: Feb 2027)",
        footer:
          "I am a data science student at Dongduk Women’s University.In this project, I served as the team leader and visualization lead, overseeing overall project planning and coordination.I was responsible for designing interactive dashboards and visualizing the effects of Official Development Assistance (ODA) on life expectancy in Ethiopia, translating complex econometric and machine-learning results into clear, policy-relevant insights for simulation and presentation.",
      },
    },
    {
      img: "/images/new_eunjun.jpg",
      name: "Choi Eunjun",
      desc: {
        intro: "Data Analysis & Modeling Researcher",
        body: "Department of Data Science,\nDongduk Women's University (Senior Year)",
        footer:
          "I am a senior student majoring in Data Science at Dongduk Women’s University. In this project, I was responsible for analyzing data and conducting simulations using econometric and machine-learning models to study the relationship between Official Development Assistance (ODA) and life expectancy.\nBased on this research, I quantitatively evaluated how development aid contributes to improvements in life expectancy.",
      },
    },
    {
      img: "/images/new_hyewon.jpg",
      name: "Yang Hyewon",
      desc: {
        intro: "Data Analysis & Modeling Researcher",
        body: "Department of Data Science & Department of Computer Science,\nDongduk Women’s University (Senior Year)",
        footer:
          "I am a senior student double majoring in Data Science and Computer Science at Dongduk Women’s University.In this project, I was responsible for analyzing the relationship between Official Development Assistance (ODA) and development outcomes through time-lagged correlation and predictive modeling.",
      },
    },
  ];

  return (
    <A.Container>
      <A.Blur1 />
      <A.Blur2 />
      <A.Nav>
        <A.Logo>
          <A.CYC>
            <img
              src={`${process.env.PUBLIC_URL}/images/logo_c.png`}
              alt="logo"
            />
            <div>CYClone</div>
          </A.CYC>
          <A.UNDP>
            <img
              src={`${process.env.PUBLIC_URL}/images/logo_u.png`}
              alt="logo"
            />
            <div>UNDP</div>
          </A.UNDP>
        </A.Logo>
        <A.Comp>
          <A.Ele style={{ cursor: "pointer" }} onClick={goHome}>
            <img
              src={`${process.env.PUBLIC_URL}/images/home_w.svg`}
              alt="home"
            />
            <div>Home</div>
          </A.Ele>
          <A.Ele style={{ cursor: "pointer" }} onClick={goProject}>
            <img
              src={`${process.env.PUBLIC_URL}/images/project_w.svg`}
              alt="home"
            />
            <div>Project</div>
          </A.Ele>
          <A.Ele style={{ cursor: "pointer" }} onClick={goModel}>
            <img
              src={`${process.env.PUBLIC_URL}/images/model_w.svg`}
              alt="home"
            />
            <div>Model</div>
          </A.Ele>
          <A.Ele style={{ cursor: "pointer" }} onClick={goAbout}>
            <img
              src={`${process.env.PUBLIC_URL}/images/about_c.svg`}
              alt="home"
            />
            <div style={{ color: "#6BB1FF" }}>About us</div>
          </A.Ele>
        </A.Comp>
      </A.Nav>

      <A.Title>Introduction</A.Title>
      <A.Det_70>
        Our team participated in the{" "}
        <Highlight style={{ fontWeight: "700" }}>2025 Data Dive for Development Hackathon</Highlight>{" "}
        and was selected as one of the two{" "}
        <Highlight style={{ fontWeight: "700" }}>Winner teams.</Highlight>
      </A.Det_70>
      <A.Det_40>
        Following the competition, we conducted an{" "}
        <Highlight style={{ fontWeight: "700" }}>
          8-week development phase
        </Highlight>{" "}
        supported by UNDP,
        <br />
        deepening our analytical approach and building a functional prototype
        that connects data insights
        <br />
        to realistic development decisions.
      </A.Det_40>

      <A.Semi>About Our Work</A.Semi>
      <A.Det_70>
        Our project focuses on analyzing{" "}
        <strong>ODA (Official Development Assistance) effectiveness</strong> and
        <br />
        identifying the conditions under which ODA investments most strongly
        contribute to development
        <br />
        outcomes—especially those related to{" "}
        <strong>life expectancy improvement.</strong>
      </A.Det_70>
      <A.Det_70>
        At the beginning of the project, we worked on structuring extensive raw
        ODA datasets and identifying
        <br />
        patterns across countries, sectors, and performance indicators. We
        refined key research questions and
        <br />
        explored pathways to produce meaningful insight for development
        strategy.
      </A.Det_70>
      <A.Det_70>
        During the mid-phase, we divided internal roles and worked in parallel
        across <strong>data analytics, story</strong>
        <br />
        <strong>development, and visualization design.</strong> We investigated
        relationships among performance indicators
        <br />
        and assessed when and where ODA investment delivers greater impact,
        using both qualitative
        <br />
        reasoning and quantitative evidence.
      </A.Det_70>
      <A.Det_70>
        In the final phase, we designed a{" "}
        <strong>visual dashboard and interactive simulation tool,</strong>{" "}
        integrating
        <br />
        analysis results to form a coherent narrative. Through internal reviews
        and external feedback loops,
        <br />
        we strengthened completeness and usability.
      </A.Det_70>

      <A.Det_99>
        Throughout the project period, we continuously managed :
      </A.Det_99>
      <A.List>
        <A.Box>Team communication and decision-making</A.Box>
        <A.Box>Weekly meetings and role adjustments</A.Box>
        <A.Box>Feedback integration and revisions</A.Box>
        <A.Box>Presentation and documentation work</A.Box>
      </A.List>

      <A.B1>
        We also engaged in iterative discussions{" "}
        <strong>with UNDP experts</strong>, applying professional guidance
        directly
        <br />
        into model refinement and result interpretation.
      </A.B1>
      <A.B2>
        Our work demonstrates not only technical capability but also effective
        teamwork and problem-solving,
        <br />
        completing the project with real-world application value.
      </A.B2>

      <A.Semi>About us</A.Semi>
      <A.Profile>
        {profiles.map((p, i) => (
          <A.Card
            key={i}
            onMouseEnter={() => setActiveIndex(i)}
            onMouseLeave={() => setActiveIndex(null)}
            active={activeIndex === i}
            shrink={activeIndex !== null && activeIndex !== i}
          >
            <A.Prof>
              <img
                src={p.img}
                alt={p.name}
                active={activeIndex === i}
                shrink={activeIndex !== null && activeIndex !== i}
              />
              <div>{p.name}</div>
            </A.Prof>

            {/* 상세정보 텍스트 */}
            <A.DetailText
              active={activeIndex === i}
              style={{ whiteSpace: "pre-line" }}
            >
              <p id="intro">{p.desc.intro}</p>
              <p id="body">{p.desc.body}</p>
              <p id="footer">{p.desc.footer}</p>
            </A.DetailText>

            {/* 오버레이 */}
            <A.Overlay active={activeIndex === i} />
          </A.Card>
        ))}
      </A.Profile>
    </A.Container>
  );
};

export default About;
