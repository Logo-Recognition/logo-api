#virtual env

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
#save new pip
pip freeze > requirements.txt

cp .env.example .env
