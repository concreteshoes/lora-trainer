#!/usr/bin/env bash

# Determine which branch to clone based on environment variables
BRANCH="master" # Default branch

if [ "$is_dev" == "true" ]; then
    BRANCH="experimental"
    echo "Development mode enabled. Cloning experimental branch..."
elif [ -n "$git_branch" ]; then
    BRANCH="$git_branch"
    echo "Custom branch specified: $git_branch"
else
    echo "Using default branch: master"
fi

# Export environment variables
extract_env() {
    local pattern="$1"

    while IFS='=' read -r key value; do
        case "$key" in
            $pattern)
                export "$key=$value"
                ;;
        esac
    done < <(tr '\0' '\n' < /proc/1/environ)
}

extract_env "GEMINI_API_KEY|HUGGING_FACE_TOKEN|SSH_PUBLIC_KEY"

# Clean up any previous failed attempts
rm -rf /tmp/lora-trainer
# Clone the repository to a temporary location with the specified branch
echo "Cloning branch '$BRANCH' from repository..."
git clone --branch "$BRANCH" https://github.com/concreteshoes/lora-trainer.git /tmp/lora-trainer

# Check if clone was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to clone branch '$BRANCH'. Falling back to main branch..."
    git clone https://github.com/concrete-shoes/lora-trainer.git /tmp/lora-trainer

    if [ $? -ne 0 ]; then
        echo "Error: Failed to clone repository. Exiting..."
        exit 1
    fi
fi

# Move start.sh to root and execute it
mv /tmp/lora-trainer/src/start.sh /
chmod +x /start.sh
bash /start.sh
