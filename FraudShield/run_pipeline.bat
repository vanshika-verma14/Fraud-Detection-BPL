@echo off
echo === FraudShield Pipeline ===
echo [1/4] Preprocessing...
python src/preprocess.py
if errorlevel 1 goto :error
echo [2/4] Training all 5 models...
python src/train.py
if errorlevel 1 goto :error
echo [3/4] Evaluating, comparing, robustness test...
python src/evaluate.py
if errorlevel 1 goto :error
echo [4/4] SHAP explainability on best model...
python src/explain.py
if errorlevel 1 goto :error
echo.
echo === Done. Launch demo: streamlit run app/streamlit_app.py ===
goto :end

:error
echo.
echo === PIPELINE FAILED. See error above. ===
exit /b 1

:end
