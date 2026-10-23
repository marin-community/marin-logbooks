# Working with research branches

Each task has an issue in `marin-community/marin-logbooks`, a research branch
derived from Marin, and `logbooks/<issue-number>-<topic>/notebook.md` in that
branch. `main` holds the index and templates. Research branches preserve the
source tree used for each experiment; they are never merged into logbooks
`main`.

## Start a task

Create or select the coordinating issue first. Describe the question, baseline,
constraints, and evidence needed to decide. Link related Marin issues. Use its
issue number and a short topic below. These commands use a new directory;
substitute the task's actual values for `123` and `rl-curriculum`.

```bash
task=123-rl-curriculum
branch="codex/$task"
task_dir="logbooks/$task"
git init "research-$task"
cd "research-$task"
git remote add marin https://github.com/marin-community/marin.git
git remote add logbooks https://github.com/marin-community/marin-logbooks.git
git fetch --no-tags marin refs/heads/main:refs/remotes/marin/main
git fetch --no-tags logbooks refs/heads/main:refs/remotes/logbooks/main
marin_base=$(git rev-parse refs/remotes/marin/main)
archive_template_revision=$(git rev-parse refs/remotes/logbooks/main)
git switch -c "$branch" "$marin_base"
mkdir -p "$task_dir"
git show "$archive_template_revision:templates/logbook.md" > "$task_dir/notebook.md"
git show "$archive_template_revision:templates/research-AGENTS.md" > "$task_dir/AGENTS.md"
git rm -r --ignore-unmatch .github/workflows
test -z "$(git ls-files .github/workflows)"
```

Record both captured revisions in the notebook frontmatter, fill in the task
issue and question, and commit the initial notebook with the archive-only
workflow removal. The removal prevents inherited automatic workflows from
running on the first push. It applies only to this research branch. These
commands assume no deliberately opted-in workflows yet.

```bash
git add "$task_dir"
git commit -m "Start research archive for #123"
git push -u logbooks "HEAD:refs/heads/$branch"
```

If that remote research branch already exists, use the join procedure below;
do not create a replacement. Check that the published branch contains the
expected notebook and workflow removal before adding links to the issue.
Record its full commit SHA with `git rev-parse HEAD`.

The research branch inherits Marin's root `AGENTS.md`, not logbooks' root
guidelines. Its directory-local `AGENTS.md` carries the archive rules. Keep that
file with the notebook and review upstream changes to these rules when needed;
the captured template revision identifies the version used.

## Join an existing task

In a new directory, initialize Git and add the two remotes as above. Then fetch
the exact task branch. Do not start a second branch for the same task.

```bash
task=123-rl-curriculum
branch="codex/$task"
task_dir="logbooks/$task"
git fetch --no-tags logbooks "refs/heads/$branch:refs/remotes/logbooks/$branch"
git switch -c "$branch" --track "logbooks/$branch"
```

Read the task issue, notebook, and directory-local `AGENTS.md` before changing
anything. Use separate local clones or worktrees for concurrent writers.

## Publish evidence and handle concurrent writers

Commit and publish experiment source and configuration before running it.
Record the resulting source SHA with the command, configuration, external
inputs, and relevant environment. Commit the notebook results, analysis, and
small figures afterward. Large artifacts remain in their existing stores.

Before each push, commit your changes, fetch the exact remote task branch, and
merge concurrent work. This keeps every published commit reachable and lets
multiple writers append to the same notebook.

```bash
git fetch --no-tags logbooks "refs/heads/$branch:refs/remotes/logbooks/$branch"
git merge --no-edit "refs/remotes/logbooks/$branch"
git push logbooks "HEAD:refs/heads/$branch"
```

If a merge conflicts, read both writers' changes and preserve both entries in
chronological order. Resolve the conflict, stage only the affected files, and
complete the merge with `git commit`. If another writer advances the branch
before your push, the push is rejected: repeat the fetch, merge, review, and
push sequence. Do not rebase published commits, reset the remote branch,
force-push, or use `--mirror`.

Only after a successful push, post the issue milestone with a commit-pinned
notebook link, for example
`https://github.com/marin-community/marin-logbooks/blob/<evidence-sha>/logbooks/123-rl-curriculum/notebook.md`.
The entry links the separate source SHA used for the experiment. Update the
issue body with the current conclusion, decisions, and open questions. A push
failure leaves the issue update pending; do not cite an unpublished local SHA.

## Bring in newer Marin code

Work from a clean research checkout. Fetch Marin explicitly, capture the new
revision, and merge it without committing so inherited workflows can be removed
before the result is published. The following block applies to research branches
with archive CI disabled.

```bash
git fetch --no-tags marin refs/heads/main:refs/remotes/marin/main
marin_update=$(git rev-parse refs/remotes/marin/main)
git merge --no-ff --no-commit "$marin_update"
git rm -rf --ignore-unmatch .github/workflows
test -z "$(git ls-files .github/workflows)"
```

The removal uses `-f` because the merge has staged incoming workflow files.
If the merge conflicts, resolve the source conflicts and workflow removals,
then review the result. If this branch already has approved archive CI, restore
its reviewed workflow definitions after removal and before committing; replace
the empty-directory check with a review that only those definitions remain.
Do not discard approved workflows or retain inherited automatic workflows.

```bash
git commit -m "Merge Marin updates into research archive"
```

Record the new Marin revision and any effects on comparisons in the notebook.
Use the concurrent-writer publish procedure before posting an issue milestone.
If the revision is already merged, there is no merge commit to create.

This procedure leaves archive CI disabled. CI setup is a separate, explicitly
requested change. GitHub requires a
[manually dispatched workflow](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
to exist on the repository's default branch; a research-branch file alone cannot
be dispatched. Review the default-branch bootstrap and selected research-branch
checks together, using `workflow_dispatch` as the sole trigger. Preserve those
approved workflows during later merges while removing inherited automatic
workflows. Do not change repository settings or credentials to enable archive CI.

## Extract the Marin contribution

Create a clean contribution branch from current Marin `main` in a separate
worktree. Inspect the research history and select the implementation, tests,
pipeline settings, and reusable fixes that Marin should maintain. Cherry-pick
appropriate source-only commits or port the selected changes; exclude notebook
material, abandoned designs, and archive-only workflow removals. Follow Marin's
normal validation and PR workflow. Do not merge the entire research branch.

Keep the research branch and exact evidence commits. Finish the notebook and
issue with the selected approach, rejected alternatives, supporting evidence,
reproduction instructions, contribution PRs, and remaining uncertainty.

## Maintain the index

Use a separate checkout of logbooks `main` to add or update a row in
[INDEX.md](INDEX.md) through a PR. Link the coordinating issue, research branch,
notebook at a published commit, and Marin contributions. Retain completed rows.
Do not merge unrelated histories or replace `main` with a Marin source tree.
