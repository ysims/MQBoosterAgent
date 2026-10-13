# Getting Started

This guide takes you through setting up Booster Studio and the MQBoosterAgent, so that you are ready to make improvements to your own soccer-playing agent.

## Booster Studio

Booster Studio is a clone of VS Code, a popular open-source IDE created by Microsoft. If you are familiar with VS Code, you can set up Booster Studio similarly to your VS Code set up, including the same extensions.

Booster Studio is created by [Booster Robotics](https://www.booster.tech/), manufacturers of humanoid robots. Booster robots have become the most popular platform in RoboCup soccer competitions, due to their ability to play soccer off-the-self, and their reasonably low cost. Some Booster Robotics employees are previous RoboCup competitors. 

To download Booster Studio:

1. Navigate to https://studio.booster.tech/

2. Click the download button for your operating system.

3. Install Booster Studio with the download file.

    On Windows, run the downloaded executable to install the program.
    On Debian-based Linux systems, use the downloaded .deb file with as package manager to install the program.
    On macOS, use the downloaded .dmg to install the program.

4. Launch Booster Studio.

This concludes the installation of Booster Studio. The next sections will set up the agent code which you can modify, and the match runner for testing.

## Get the Agent Code

If you are working in a team, one person will need to fork the MQBoosterAgent repository. This is a one-time set up. 

If you just want to try out the default agent, you can skip these steps and proceed using the https://github.com/ysims/MQBoosterAgent repository.

1. Navigate to https://github.com/ysims/MQBoosterAgent in a browser.  

2. If you are not signed in to GitHub, sign in. If you do not have an account, make a GitHub account and sign in.

3. Click the 'fork' button in the top right hand side of the MQBoosterAgent repository.

4. Choose your own account as the owner. Update the name of the repository to make it unique to your team. Click 'Create Fork'. 

5. Add your teammates to the repository, so that they can make changes. 
    1. Go to your repository settings on the far right of the top bar.
    2. Click 'Collaborators' in the sidebar. 
    3. Click 'Add People' and type in the GitHub usernames of your teammates.

Now, you and your teammates have access to a new repository based on the default MQ Booster Agent code. The following steps are for every member of the team, on any device they are setting up to run the agent with.

The following instructions use Git CLI. On macOS and Linux, use the terminal to run these commands. On Windows, install [Git Bash](https://git-scm.com/install/windows) and run the commands in the Git Bash terminal. 

You may choose to use a Git GUI such as GitHub Desktop or GitKraken instead. Refer to the documentation of those tools to clone a repository from GitHub.

1. Open a terminal.
2. Navigate to the folder where you would like to keep your agent code. An example here is to make a `code` directory and navigate into that `code` directory:

    ```
    mkdir code
    cd code
    ```

3. Clone the repository:

    ```
    git clone <url>
    ```

    Replace `<url>` with your repository URL.

The code is now on your local computer. Next, open it in Booster Studio. 

1. Open Booster Studio, if it is not already open.
2. Click File->'Open Folder'.
3. Navigate to your newly cloned folder and select it.
4. You should see the repository in the sidebar, in the explorer tab.

## Get the Match Runner Extension

These instructions follow the instructions in https://github.com/Samge0/booster-match-runner. The repository contains detailed screenshots.

1. Open Booster Studio, if it is not already open.
2. Navigate in the sidebar to the extensions tab.
3. Search for 'Booster Match Runner'.
4. Install the Booster Match Runner extension.
5. Click the sidebar drop down and pin the Match Runner extension.

## Set Up the Simulation Docker Image

A mirror of the Booster Docker image is provided, with an entrypoint change to enable vision bounding box detections.

1. Pull the Docker image:

    ```
    docker pull ysims/boostermq:0.6.5-beta
    ```

2. Tag the image so that Booster Studio finds and uses it:

    ```
    docker tag ysims/boostermq:0.6.5-beta \
        booster-robotics-registry.cn-beijing.cr.aliyuncs.com/virtual-robot/virtual-robot:0.6.5-beta
    ```

## Run a Game

Make sure the previous set up instruction have been completed with no errors. These instructions run the local agent code in a 3v3 soccer match.

1. Click the 'Activate, build, deploy and run agent' in the top right corner of Booster Studio to prepare the agent.
2. In the Match Runner, select your agent (e.g. mqagent) for both teams. The blue team may need reselecting on each new code build, otherwise the blue team container may not update. Do not use `mqagent.blue`.
3. Click 'Start match + UI'.
4. A separate window should launch, with a soccer field and a 3v3 match.

When you have changed code, it is a good idea to reset everything, as it is hard to tell if the robots in the simulator are running the new code. 

1. Click 'End' in the match runner and close the simulation window.
2. Click 'Activate, build, deploy and run agent'.
3. Reselect the blue team in the Match Runner. 
4. Click 'Start Match + UI' in the Match Runner.

