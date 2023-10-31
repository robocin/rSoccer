# Repository Guidelines

### Code changes should use the following pipeline:
- Fork the repository and make the changes on it
- Create a pull request to the dev branch
- Before merging the pull request, the changes needs to be rebased on the dev branch
- Squash merge should be used
- Document changes made on the PR, and the name of the merged pr commit should follow [this guideline](http://karma-runner.github.io/0.10/dev/git-commit-msg.html)
- main branch will should be fast forwarded to dev after dev branch is validated

Members with permissions to create branches directly should not do so for creating new environments.


## Example Pull Request Template:
```markdown
## Description:

- Description of pull request

## Changes:

- Changes implemented on pull request

## TODO:

- Tasks to be done before pull request is ready to be merged

## Future changes:

- Changes which will not be implemented on this pull request but can/should be done later

```