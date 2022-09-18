mkdir -p ~/.streamlit/
echo "[server]"  >> ~/.streamlit/config.toml
echo "headless = true"  >> ~/.streamlit/config.toml
echo "port = $PORT"  >> ~/.streamlit/config.toml
echo "enableCORS = false"  >> ~/.streamlit/config.toml
echo "[global]"  >> ~/.streamlit/config.toml
echo "developmentMode = false"  >> ~/.streamlit/config.toml
