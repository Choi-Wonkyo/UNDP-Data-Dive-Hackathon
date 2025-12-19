import React, { useEffect } from "react";
import * as M from "../styles/StyledModel";
import { useNavigate } from "react-router-dom";

const Model = () => {
  const navigate = useNavigate();

  const goProject = () => {
    navigate(`/project`);
    window.scrollTo(0, 0);
  };
  const goHome = () => navigate(`/`);
  const goAbout = () => navigate(`/about`);
  const goModel = () => navigate(`/model`);

  useEffect(() => {
    window.scrollTo(0, 0); // 페이지 맨 위로 스크롤 이동
  }, []);

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
              src={`${process.env.PUBLIC_URL}/images/home_w.svg`}
              alt="home"
            />
            <div>Home</div>
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
              src={`${process.env.PUBLIC_URL}/images/model_c.svg`}
              alt="home"
            />
            <div style={{ color: "#6BB1FF" }}>Model</div>
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

      <M.Title>Model Workflow</M.Title>
      <M.Flow1>
        <M.Set>
          <M.FTit>Data Input</M.FTit>
          <M.FDet>
            ODA amounts
            <br />
            Shock variables
            <br />
            Context variables
          </M.FDet>
        </M.Set>
        <img src={`${process.env.PUBLIC_URL}/images/go.svg`} alt="next" />
        <M.Set>
          <M.FTit>Preprocessing</M.FTit>
          <M.FDet>
            Cleaning
            <br />
            Scaling
            <br />
            Time-lag alignment
          </M.FDet>
        </M.Set>
        <img src={`${process.env.PUBLIC_URL}/images/go.svg`} alt="next" />
        <M.Set>
          <M.FTit>Model Training</M.FTit>
          <M.FDet>
            MLP single-target
            <br />
            models + SHAP
          </M.FDet>
        </M.Set>
      </M.Flow1>

      <M.Flow2>
        <img src={`${process.env.PUBLIC_URL}/images/go.svg`} alt="next" />
        <M.Set>
          <M.FTit>Scenario Prediction</M.FTit>
          <M.FDet>
            User-adjusted ODA inputs
            <br />
            -> Future indicator forecasts
          </M.FDet>
        </M.Set>
        <img src={`${process.env.PUBLIC_URL}/images/go.svg`} alt="next" />
        <M.Set>
          <M.FTit>Result Analysis</M.FTit>
          <M.FDet>
            Key drivers &<br />
            policy insights
          </M.FDet>
        </M.Set>
      </M.Flow2>

      <M.Over>Model Overview</M.Over>
      <M.Goal>Goal</M.Goal>
      <M.GDet>
        This model predicts the impact of{" "}
        <M.Highlight>ODA (Official Development Assistance)</M.Highlight>
        <br />
        and <M.Highlight>institutional quality</M.Highlight> on{" "}
        <M.Highlight>Ethiopia’s Life Expectancy,</M.Highlight> based on changes
        in <M.Highlight>ODA amounts.</M.Highlight>
        <br />
        It uses <M.Highlight>scenario-based predictions</M.Highlight> to
        estimate how changes in ODA influence life expectancy.
      </M.GDet>
      <M.User>Target Users</M.User>
      <M.UDet>
        This model helps policymakers and development agencies{" "}
        <M.Highlight>predict</M.Highlight> the impact of increasing specific
        <br />
        <M.Highlight>ODA sectors</M.Highlight> on life expectancy, and{" "}
        <M.Highlight>compare various policy scenarios</M.Highlight> based on
        those predictions.
      </M.UDet>

      <M.Descr>Key Model Description</M.Descr>
      <M.Tail>(1) Variables and Data</M.Tail>
      <M.Key>
        <div id="title">Key Input Variables</div>
        <ul id="detail">
          <li>
            <strong>Health ODA</strong> (Health Sector): Changes in
            health-related aid
          </li>
          <li>
            <strong>Social/Env ODA</strong> (Social/Environmental Sector): Aid
            related to sanitation, environment, and basic health services
          </li>
          <li>
            <strong>Gov ODA</strong> (Governance Sector): Aid aimed at improving
            administration and institutional quality
          </li>
          <li>
            <strong>RQ</strong> (Regulatory Quality): Institutional quality
            indicator (using World Governance Indicators)
          </li>
        </ul>
      </M.Key>
      <M.Model>
        <div id="title">Model Type</div>
        <ul id="detail">
          <li>
            <strong>Dynamic Linear Model (DLM)</strong> was used to analyze the{" "}
            <strong>dynamic relationship</strong> between{" "}
            <strong>ODA changes</strong> and <strong>life</strong>
            <br />
            <strong>expectancy,</strong> distinguishing between{" "}
            <strong>short-term (immediate effects) and lag effects</strong>{" "}
            (lag1, lag2).
          </li>
          <li>
            After conducting <strong>Variance Inflation Factor (VIF)</strong>{" "}
            analysis to resolve
            <strong>multicollinearity issues,</strong> only the most
            <br />
            significant variables were selected to optimize the model.
          </li>
          <li>
            Ultimately, <strong>OLS-HAC</strong> (robust standard error
            regression) was applied to build an accurate{" "}
            <strong>predictive model,</strong>
            <br />
            considering <strong>heteroskedasticity</strong> and{" "}
            <strong>autocorrelation.</strong>
          </li>
        </ul>
      </M.Model>

      <M.Tail>(2) Model’s Key Points</M.Tail>
      <M.TBox>
        <ul id="detail">
          <li>
            <strong>Immediate Effects</strong> (Horizon 0): Changes in{" "}
            <strong>Social/Env ODA</strong> and
            <strong>Infrastructure, Education</strong> are{" "}
            <strong>immediately</strong>
            <br />
            reflected in life expectancy.
          </li>
          <li>
            <strong>Lag Effects</strong> (Horizon 1, 2):{" "}
            <strong>Health ODA</strong> shows{" "}
            <strong>2+ years of lag effects</strong>
            due to infrastructure development and
            <br />
            health service improvements.
          </li>
          <li>
            <strong>Coefficients and Predictions</strong>: The model uses{" "}
            <strong>LASSO regression</strong> to select important variables and{" "}
            <strong>HAC</strong>
            <br />
            covariance estimation to provide predictions that consider
            <strong> uncentainty.</strong>
          </li>
        </ul>
      </M.TBox>

      <M.Dash>Dashboard Explanation</M.Dash>
      <M.T1>(1) Dashboard Scenario Adjustment</M.T1>
      <M.B1>
        <div id="title">
          Users can set the following policy scenarios in the dashboard:
        </div>
        <ul id="detail">
          <li>
            <strong>Set change rates for each ODA sector</strong>
          </li>
          <div id="ex1" style={{ marginBottom: "20px" }}>
            : ex) Increase <strong>Health ODA </strong>
            by <strong>+10%</strong>, and <strong>Social/Env ODA</strong> by{" "}
            <strong>+5%.</strong>
          </div>
          <li>
            <strong>Set change rate for RQ</strong>
          </li>
          <div id="ex1">
            : Adjust the rate of change for{" "}
            <strong>institutional quality</strong> (Regulatory Quality) to reflect improvements
            in governance.
          </div>
        </ul>
      </M.B1>

      <M.T1>(2) Scenario Results and Reflections</M.T1>
      <M.B2>
        <ul id="detail">
          <li>
            <strong>Immediate Effects (Horizon 0)</strong>
          </li>
          <div style={{ marginBottom: "30px" }}>
            : The <strong>immediate effects</strong> of each set ODA change rate
            are displayed in graphs and numbers. For example, a{" "}
            <strong>10% increase in Social/Env ODA</strong> will immediately
            increase life expectancy by <strong>+0.05 years.</strong>
          </div>
          <li>
            <strong>Lag Effects (Horizon 1 & 2)</strong>
          </li>
          <div>
            : For <strong>Health ODA</strong>, the{" "}
            <strong>cumulative effects</strong>
            of <strong>1 year</strong> and <strong>2 years</strong> are shown
            visually, helping to understand the
            <br />
            <strong>long-term impact</strong> of policies.
          </div>
        </ul>
      </M.B2>

      <M.T1>(3) Example Result Interpretation (Based on 2023 Data)</M.T1>
      <M.B3>
        <ul>
          <li>
            <strong>Health ODA +10%</strong>: +0.03 years increase in{" "}
            <strong>2024</strong>, +0.10 years increase in{" "}
            <strong>2025 → +0.13 years</strong> after 2 years.
          </li>
          <li>
            <strong>Social/Env ODA +10%</strong>: Immediate{" "}
            <strong>+0.05 years</strong> increase.
          </li>
        </ul>
        <div>
          The <strong>immediate effects</strong> and{" "}
          <strong>cumulative effects</strong> can be intuitively compared over
          time.
        </div>
      </M.B3>

      <M.Example>Policy Scenario Examples</M.Example>
      <M.T1>(1) Example 1: Health ODA 10% Increase</M.T1>
      <M.B4>
        <ul>
          <li>
            <strong>Assumption</strong>: Increase in Health ODA by 10%
          </li>
          <li>
            2024: ΔLifeExp ≈ <strong>+0.03 years</strong> (short-term response)
          </li>
          <li>
            2025: ΔLifeExp ≈ <strong>+0.10 years</strong> (2-year response)
          </li>
          <li>
            <strong>2-year cumulative effect</strong>: Approx{" "}
            <strong>+0.13 years</strong> (about 1.6 months increase)
          </li>
        </ul>
      </M.B4>

      <M.T1>
        (2) Example 2: Health ODA 10% Increase, Social/Env ODA 5% Increase
      </M.T1>
      <M.B5>
        <ul>
          <li>
            <strong>Assumption</strong>: Increase Health ODA by 10%, Social/Env
            ODA by 5%
          </li>
          <li>
            <strong>2024</strong>: ΔLifeExp ≈ <strong>+0.05 years</strong>{" "}
            (immediate effect from Social/Env ODA)
          </li>
          <li>
            <strong>2025</strong>: ΔLifeExp ≈ <strong>+0.03 years</strong>{" "}
            (Health ODA 1-year lag effect)
          </li>
          <li>
            <strong>2026</strong>: ΔLifeExp ≈ <strong>+0.10 years</strong>{" "}
            (Health ODA 2-year lag effect)
          </li>
          <li>
            <strong>2-year cumulative effect</strong>: Approx{" "}
            <strong>+0.18~0.20 years</strong> (about 2-3 months increase)
          </li>
        </ul>
      </M.B5>

      <M.T1>Conclusion</M.T1>
      <M.B6>
        <div>
          <strong>“Health ODA</strong> and <strong>Social/Env ODA</strong>{" "}
          combined result in not only{" "}
          <strong>short-term life expectancy increases</strong>, but also a
          significant <strong>long-term health improvement effect</strong>{" "}
          driven by <strong>Health ODA’s lag effect</strong> after 2 years. The{" "}
          <strong>2-year cumulative effect</strong> shows{" "}
          <strong>+0.13 years</strong> for Health ODA alone, with
          <strong>+0.18~0.20 years</strong> from the combined scenario.”
        </div>
      </M.B6>

      <M.Limit>Model Usage and Limitations</M.Limit>
      <M.T1>(1) Model Usage</M.T1>
      <M.B7>
        <ul>
          <li>
            <strong>Policy Scenario Adjustments</strong> allow users to{" "}
            <strong>real-time predictions</strong> of life expectancy changes
            based on
            <br />
            increases in specific <strong>ODA sectors.</strong>
          </li>
          <li>
            <strong>"What-If Simulation"</strong>: Users can{" "}
            <strong>analyze</strong> the expected impact over the
            <strong>next 1-2 years</strong> from various policy
            <br />
            scenarios in an intuitive way.
          </li>
        </ul>
      </M.B7>

      <M.T1>(2) Model Limitations</M.T1>
      <M.B8>
        <ul>
          <li>
            <strong>External Shocks</strong> (e.g., pandemics) or{" "}
            <strong>unforeseen economic changes</strong> are not accounted for
            in the model.
          </li>
          <li>
            Additional variables need to be considered or{" "}
            <strong>updated estimates</strong>
            should be applied.
          </li>
          <li>
            <strong>Changes in Regulatory Quality</strong> (RQ) may take time to
            reflect in policy outcomes, making it challenging to
            <br />
            predict <strong>short-term changes.</strong>
          </li>
        </ul>
      </M.B8>

      <M.T1>(3) Future Model Enhancements</M.T1>
      <M.B9>
        <ul>
          <li>Shock and Normal Period Setup</li>
          <li>External Variables and Indicators Integration</li>
          <li>Time-Series Model Enhancements</li>
        </ul>
      </M.B9>

      <M.Button
        style={{ cursor: "pointer" }}
        onClick={() => window.open("https://cyclone-undpproject.streamlit.app/")}
      >
        Run Simulation
      </M.Button>
    </M.Container>
  );
};

export default Model;
