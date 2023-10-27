#!/bin/bash
# Usage:
#  ./scripts/publish_pip.sh [version_type]
#
# Args:
#  version_type: Default: patch. The type of version bump to do. The version should be a valid poetry semver version
#  type, taken from: https://python-poetry.org/docs/cli/#version.
#
# Important!
# Make sure you have `gh` installed and are logged in with:
#  $ brew install gh
#  $ gh auth login

set -e # Fail if any of the commands below fail.

VERSION_TYPE="$1"
if [[ $VERSION_TYPE == "" ]]; then
  VERSION_TYPE="patch"
fi

# Make sure the version type argument is valid.
echo "Dry run bumping version..."
poetry version --dry-run $VERSION_TYPE

# Make sure the user is logged into github and has `gh` installed.
echo "Checking github auth..."
gh auth status || exit 1

# Make sure the user is logged in and can see the cloud builds.
echo "Checking Google Cloud authentication..."
gcloud builds list --project lilac-386213 || exit 1

set -o allexport
source .env.local
set +o allexport

if [[ -z "${PYPI_TOKEN}" ]]; then
  echo 'Please set the PYPI_TOKEN variable in your .env.local file'
  exit 1
fi

CHANGES=`git status --porcelain`
CUR_BRANCH=`git rev-parse --abbrev-ref HEAD`

if [[ "$CUR_BRANCH" != "main" ]]; then
  echo "Please checkout the main branch before publishing."
  exit 1
fi

if [ ! -z "$CHANGES" ];
then
  echo "Make sure the main branch has no changes. Found changes:"
  echo $CHANGES
  exit 1
fi

# Build the web server.
echo "Building webserver..."
./scripts/build_server_prod.sh

# Upgrade the version in pyproject.
poetry version $VERSION_TYPE

NEW_VERSION=`poetry version --short`
TAG_ID="v$NEW_VERSION"

# PYPI_TOKEN must be set in .env.local.
poetry config pypi-token.pypi $PYPI_TOKEN

# Build the wheel file.
echo "Building wheel..."
rm -rf dist/*
poetry build

# Publish to pip.

read -p "Continue (y/n)? " CONT
if [ "$CONT" = "y" ]; then
  poetry publish
  echo "Published $(poetry version)"
else
  echo "Did not publish"
  exit 1
fi

# Commit directly to main.
echo "Commiting version in pyproject.toml to main..."
git commit -a -m "Bump version to $NEW_VERSION."
git push origin main

# Create the tag and push it.
echo "Tagging with $TAG_ID..."
git tag "$TAG_ID" && git push --tags

# Sleep for 2 seconds because the release script fails if the tag is not found.
sleep 2

# Create the release with auto-generated release notes.
echo "Creating release..."
gh release create "$TAG_ID" ./dist/*.whl \
  --generate-notes \
  --latest \
  --title "$TAG_ID" \
  --verify-tag

# Building the docker images.
gcloud builds submit \
  --config cloudbuild.yml \
  --substitutions=_VERSION=$(poetry version -s) \
  --async .
