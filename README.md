![banner image](https://images.aicrowd.com/raw_images/challenges/banner_file/1135/6c90e052800987c29365.png)
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/yWurtB2huX)

# Meta KDD Cup '24 [CRAG: Comprehensive RAG Benchmark](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024) Starter Kit


This repository is the CRAG: Comphrensive RAG Benchmark **Submission template and Starter kit**! Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your models to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your model, etc.
*  **Starter code** for you to get started!

# Table of Contents

1. [Competition Overview](#-competition-overview)
2. [Dataset](#-dataset)
3. [Tasks](#-tasks)
4. [Evaluation Metrics](#-evaluation-metrics)
5. [Getting Started](#-getting-started)
   - [How to write your own model?](#️-how-to-write-your-own-model)
   - [How to start participating?](#-how-to-start-participating)
      - [Setup](#setup)
      - [How to make a submission?](#-how-to-make-a-submission)
      - [What hardware does my code run on?](#-what-hardware-does-my-code-run-on-)
      - [How are my model responses parsed by the evaluators?](#-how-are-my-model-responses-parsed-by-the-evaluators-)
6. [Frequently Asked Questions](#-frequently-asked-questions)
6. [Important Links](#-important-links)


# 📖 Competition Overview


# 📊 Dataset


# 👨‍💻👩‍💻 Tasks  


## 📏 Evaluation Metrics


Please refer to [local_evaluation.py](local_evaluation.py) for more details on how we will evaluate your submissions.

# 🏁 Getting Started
1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024).
2. **Fork** this starter kit repository. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/forks/new) to create a fork.
3. **Clone** your forked repo and start developing your model.
4. **Develop** your model(s) following the template in [how to write your own model](#how-to-write-your-own-model) section.
5. [**Submit**](#-how-to-make-a-submission) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation [(full instructions below)](#-how-to-make-a-submission). The automated evaluation setup will evaluate the submissions on the private datasets and report the metrics on the leaderboard of the competition.

# ✍️ How to write your own model?

Please follow the instructions in [models/README.md](models/README.md) for instructions and examples on how to write your own models for this competition.

# 🚴 How to start participating?

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/profile/keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/user/ssh.html).




2. **Fork the repository**. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit/-/forks/new) to create a fork.

2.  **Clone the repository**

    ```bash
    git clone git@gitlab.aicrowd.com:aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/meta-comphrehensive-rag-benchmark-starter-kit.git
    cd meta-comphrehensive-rag-benchmark-starter-kit
    ```

3. **Install** competition specific dependencies!
    ```bash
    cd meta-comphrehensive-rag-benchmark-starter-kit
    pip install -r requirements.txt
    ```

4. Write your own model as described in [How to write your own model](#how-to-write-your-own-model) section.

5. Test your model locally using `python local_evaluation.py`.

6. Accept the Challenge Rules on the main [challenge page](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024) by clicking on the **Participate** button. Also accept the Challenge Rules on the Task specific page (link on the challenge page) that you want to submit to.

7. Make a submission as described in [How to make a submission](#-how-to-make-a-submission) section.


## 📮 How to make a submission?

Please follow the instructions in [docs/submission.md](docs/submission.md) to make your first submission. 
This also includes instructions on [specifying your software runtime](docs/submission.md#specifying-software-runtime-and-dependencies), [code structure](docs/submission.md#code-structure-guidelines), [submitting to different tracks](docs/submission.md#submitting-to-different-tracks).

**Note**: **Remember to accept the Challenge Rules** on the challenge page, **and** the task page before making your first submission.

## 💻 What hardware does my code run on ?
You can find more details about the hardware and system configuration in [docs/hardware-and-system-config.md](docs/hardware-and-system-config.md).
In summary, we provide you `4` x [[NVIDIA T4 GPUs](https://www.nvidia.com/en-us/data-center/tesla-t4/)].


# ❓ Frequently Asked Questions
## Which track is this starter kit for ?
This starter kit can be used to submit to any of the tracks. You can find more information in [docs/submission.md#submitting-to-different-tracks](docs/submission.md#submitting-to-different-tracks).

**Best of Luck** :tada: :tada:

# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024
- 🗣 Discussion Forum: https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/discussion
- 🏆 Leaderboard: https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/leaderboards
