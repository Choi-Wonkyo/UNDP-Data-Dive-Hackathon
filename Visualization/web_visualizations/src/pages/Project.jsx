import React from "react";
import * as P from "../styles/StyledProject";
import { useNavigate } from "react-router-dom";

const Project = () => {
  const navigate = useNavigate();

  const goProject = () => navigate(`/project`);
  const goHome = () => navigate(`/`);
  const goAbout = () => navigate(`/about`);
  const goModel = () => navigate(`/model`);

  return (
    <P.Container>
      <P.Nav>
        <P.Logo>
          <P.CYC>
            <img
              src={`${process.env.PUBLIC_URL}/images/logo_c.png`}
              alt="logo"
            />
            <div>CYClone</div>
          </P.CYC>
          <P.UNDP>
            <img
              src={`${process.env.PUBLIC_URL}/images/logo_u.png`}
              alt="logo"
            />
            <div>UNDP</div>
          </P.UNDP>
        </P.Logo>
        <P.Comp>
          <P.Ele style={{ cursor: "pointer" }} onClick={goHome}>
            <img
              src={`${process.env.PUBLIC_URL}/images/home_w.svg`}
              alt="home"
            />
            <div>Home</div>
          </P.Ele>
          <P.Ele style={{ cursor: "pointer" }} onClick={goProject}>
            <img
              src={`${process.env.PUBLIC_URL}/images/project_c.svg`}
              alt="home"
            />
            <div style={{ color: "#6BB1FF" }}>Project</div>
          </P.Ele>
          <P.Ele style={{ cursor: "pointer" }} onClick={goModel}>
            <img
              src={`${process.env.PUBLIC_URL}/images/model_w.svg`}
              alt="home"
            />
            <div>Model</div>
          </P.Ele>
          <P.Ele style={{ cursor: "pointer" }} onClick={goAbout}>
            <img
              src={`${process.env.PUBLIC_URL}/images/about_w.svg`}
              alt="home"
            />
            <div>About us</div>
          </P.Ele>
        </P.Comp>
      </P.Nav>

      <P.Title>ODA x Life Expectancy Insights</P.Title>
      <P.Sub>
        LifeSim is a data-driven platform simulating how ODA interventions
        <br />
        influence life expectancy and socio-economic development in Ethiopia,
        <br />
        enabling scenario-based policy insights for sustainable development
        planning.
      </P.Sub>

      <P.Tit>Our Methodology</P.Tit>
      <P.Det>
        We integrate historical ODA data, socio-economic indicators,
        <br />
        and governance variables for Ethiopia (2014–2021) to estimate ODA’s
        impact on life expectancy.
      </P.Det>
      <P.Det>
        Using a multi-lag Dynamic Linear Model (DLM) with LASSO variable
        selection
        <br />
        and OLS-HAC estimation, we capture both immediate and delayed effects
        across sectors
        <br />
        such as Health, Education, Infrastructure, Governance, and
        Social/Environmental.
      </P.Det>
      <P.Det>
        Scenario-based simulations and sensitivity analyses allow us to quantify
        short-, medium-,
        <br />
        and long-term impacts, identify optimal ODA combinations, and generate
        actionable policy insights.
      </P.Det>

      <P.List>
        <P.Box>
          <P.Img1 />
          <P.Img3 />
          <P.Img2 />
          <P.Img4 />
          <P.BTit>Why Ethiopia?</P.BTit>
          <P.BDet>
            <div>
              Ethiopia has received significant ODA over the past decades, with
              gradual changes across multiple development sectors.
            </div>
            <div>
              The country presents diverse economic, health, and education
              indicators, making it suitable for policy simulation analysis.
            </div>
            <div>
              It serves as a practical case for policymakers and international
              development organizations to apply insights from simulations.
            </div>
            
            <br />

            <div style={{ fontSize: "0.85rem", opacity: 0.7 }}>
              Data Sources: CRS ODA Data (UNDP – Seoul Policy Centre);
              World Bank Open Data – World Development Indicators
            </div>
            
          </P.BDet>
        </P.Box>
        <P.Box>
          <P.Blur1 />
          <P.Blur2 />
          <P.Blur3 />
          <P.Blur4 />
          <P.BTit>Why Life Expectancy?</P.BTit>
          <P.BDet>
            <div>
              Life expectancy is a key development indicator reflecting health,
              living conditions, and socio-economic factors.
            </div>
            <div>
              It intuitively shows the impact of ODA investments and policy
              changes on people’s quality of life.
            </div>
            <div>
              Although a single metric, its correlation with other
              socio-economic indicators captures broader development outcomes.
            </div>

            <br />

            <div style={{ fontSize: "0.85rem", opacity: 0.7 }}>
              Reference: International Journal of Health Planning and Management, Vol. 33,
              No. 1 (2018)
            </div>
            
          </P.BDet>
        </P.Box>
        <P.Box>
          <P.G1 />
          <P.G2 />
          <P.G3 />
          <P.G4 />
          <P.BTit>Why Scenario-Based Modeling?</P.BTit>
          <P.BDet style={{ gap: "40px", marginTop: "93px" }}>
            <div>
              Simple time series models are limited to extrapolating past trends
              and cannot simultaneously
              <br />
              account for multiple influencing factors such as sectoral ODA
              investments, external shocks, and governance quality.
            </div>
            <div>
              Our scenario-based DLM framework enables testing “what-if” policy
              interventions, capturing immediate and delayed effects,
              <br />
              and identifying optimal ODA combinations — insights that basic
              time series models cannot provide.
            </div>
          </P.BDet>
        </P.Box>
      </P.List>



      
      <P.Conclusion>Conclusion</P.Conclusion>
      <P.Con>
        <div style={{ fontSize: "1.1rem", fontWeight: 1000 }}>
          LifeSim: Evidence-Based Insights for ODA Optimization
        </div>
        <div>
          LifeSim is a data-driven framework that explains how ODA investments affect life expectancy in Ethiopia
          <br />
          across key sectors, accounting for time-lagged effects. 
        </div>
        
        <br>
        </br>
        
        <div style={{ fontSize: "1.1rem"}}>
          <span style={{ fontWeight: 700 }}>🔍 Model Overview</span><br />
          <span style={{ fontWeight: 400 }}>
            Method: Dynamic Linear Model (DLM) with OLS-HAC 
            <br />
            Performance: R² = 0.851
            <br />
            Explanatory Power: ~76% of life expectancy variation explained
            <br />
             → High confidence in policy-relevant insights
          </span>
        </div>

        <br>
        </br>
        
        <div style={{ fontSize: "1.1rem", fontWeight: 1000 }}>
          📊 Key Sectoral Findings
        </div>
        <div style={{fontSize: "1.1rem"}}>
          🏥 Health ODA
          <br />
          Effect: Strong positive impact (+1.08) 
          <br />
          Lag: 2 years
          <br />
          Interpretation: Benefits emerge after infrastructure and service expansion
        </div>
        
        <div style={{fontSize: "1.1rem"}}>
          🌱 Social & Environmental ODA
          <br />
          Effect: Positive impact (+0.52)
          <br />
          Lag: Immediate (Lag 0)
          <br />
          Interpretation: Quick improvements in living conditions
        </div>

        <div style={{fontSize: "1.1rem"}}>
          🏛 Governance ODA
          <br />
          Effect: Short-term negative impact (−0.41)
          <br />
          Lag: 1 year
          <br />
          Interpretation: Temporary inefficiencies from institutional restructuring
        </div>

        <br>
        </br>

        <div style={{ fontSize: "1.1rem"}}>
          <span style={{ fontWeight: 700 }}>🧩 Institutional Amplifier</span><br />
          <span style={{ fontWeight: 400 }}>
            Regulatory Quality (RQ) strengthens both the magnitude and durability of ODA effects across sectors.
          </span>
        </div>

        <div style={{ fontSize: "1.1rem"}}>
          <span style={{ fontWeight: 700 }}>🎯 Takeaway for Policymakers</span><br />
          <span style={{ fontWeight: 400 }}>
            ODA effectiveness depends on sector-specific time horizons
            <br />
            Short-term outcomes and long-term gains must be planned differently
            <br />
            Institutional quality is critical for sustainable impact
          </span>
        </div>
      </P.Con>



      <P.Github
        style={{ cursor: "pointer" }}
        onClick={() =>
          window.open("https://github.com/CYClone-22/undp-odaproj/tree/main")
        }
      >
        <img src={`${process.env.PUBLIC_URL}/images/github.png`} alt="github" />
        <P.GDet>
          <div id="text">More information on</div>
          <div id="bold"> GitHub</div>
        </P.GDet>
      </P.Github>
    </P.Container>
  );
};

export default Project;
