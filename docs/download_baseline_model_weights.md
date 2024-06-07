### Setting Up and Downloading Baseline Model weighta with Hugging Face

This guide outlines the steps to download (and check in) the models weights required for the baseline models.
We will focus on the `Meta-Llama-3-8B-Instruct` and `all-MiniLM-L6-v2` models.
But the steps should work equally well for any other models on hugging face. 

#### Preliminary Steps:

1. **Install the Hugging Face Hub Package**:
   
   Begin by installing the `huggingface_hub` package, which includes the `hf_transfer` utility, by running the following command in your terminal:

   ```bash
   pip install huggingface_hub[hf_transfer]
   ```

2. **Accept the LLaMA Terms**:
   
   You must accept the LLaMA model's terms of use by visiting: [meta-llama/Meta-Llama-3-8B-Instruct Terms](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct).

3. **Create a Hugging Face CLI Token**:
   
   Generate a CLI token by navigating to: [Hugging Face Token Settings](https://huggingface.co/settings/tokens). You will need this token for authentication.

#### Hugging Face Authentication:

1. **Login via CLI**:
   
   Authenticate yourself with the Hugging Face CLI using the token created in the previous step. Run:

   ```bash
   huggingface-cli login
   ```

   When prompted, enter the token.

#### Model Downloads:

1. **Download LLaMA-3-8B-Instruct Model**:

   Execute the following command to download the `Meta-Llama-3-8B-Instruct` model to a local subdirectory. This command excludes unnecessary files to save space:

   ```bash
   HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
       meta-llama/Meta-Llama-3-8B-Instruct \
       --local-dir-use-symlinks False \
       --local-dir models/meta-llama/Meta-Llama-3-8B-Instruct \
       --exclude *.pth # These are alternates to the safetensors hence not needed
   ```

3. **Download MiniLM-L6-v2 Model (for sentence embeddings)**:

   Similarly, download the `sentence-transformers/all-MiniLM-L6-v2` model using the following command:

   ```bash
   HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
      sentence-transformers/all-MiniLM-L6-v2 \
       --local-dir-use-symlinks False \
       --local-dir models/sentence-transformers/all-MiniLM-L6-v2 \
       --exclude *.bin *.h5 *.ot # These are alternates to the safetensors hence not needed
   ```


#### Version Control with Git LFS:

1. **Track Model Weights**:
   
   Use Git Large File Storage (LFS) to track the model directories. This ensures efficient handling of large files:

   ```bash
   git lfs track "models/meta-llama/*"
   git lfs track "models/sentence-transformers/*"
   ```

2. **Commit and Push**:
   
   Add the models to your Git repository, commit the changes, and push them to your remote repository:

   ```bash
   git add models/
   git commit -am "add weights"
   git push origin master
   ```
If you are struggling with GIT-LFS, you are very much encouraged to check out [this post](https://discourse.aicrowd.com/t/how-to-upload-large-files-size-to-your-submission/2304).
