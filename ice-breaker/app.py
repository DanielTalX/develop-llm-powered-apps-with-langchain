from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from ice_breaker import ice_break_with
from utils.llm_utils import get_llm_model, LLMModel, LlmModelName

load_dotenv()

app = Flask(__name__)

llm: LLMModel = get_llm_model(model_name=LlmModelName.LLAMA_3_1)
linkedin_profile_url_real = "https://www.linkedin.com/in/eden-marco/"
linkedin_profile_url_gist = "https://gist.githubusercontent.com/emarco177/859ec7d786b45d8e3e3f688c6c9139d8/raw/32f3c85b9513994c572613f2c8b376b633bfc43f/eden-marco-scrapin.json"
EDEN_TWITTER_GIST = "https://gist.githubusercontent.com/emarco177/827323bb599553d0f0e662da07b9ff68/raw/57bf38cf8acce0c87e060f9bb51f6ab72098fbd6/eden-marco-twitter.json"

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    name = request.form["name"]

    summary, profile_pic_url = ice_break_with(
        llm=llm,
        name_of_person=name,
        mock_linkedin=True,
        mock_twitter=True,
        twitter_gist_url=EDEN_TWITTER_GIST)

    return jsonify(
        {
            "summary_and_facts": summary.to_dict(),
            "picture_url": profile_pic_url,
        }
    )


if __name__ == "__main__":

    app.run(host="0.0.0.0", port=3010, debug=True)