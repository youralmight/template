
These are the steps:  
1. Try to build container based on some configuration file, and get stuck.
2. Copy vscode-server from "~/.vscode-server/bin/CURRENT-VERSION" in the host to "/root/.vscode-server/bin/CURRENT-VERSION" in the running container with `docker cp` command assited by `tar` or `zip`.
<!-- 3. [Optional] Get download link for vscode-server standalone, namely cli, from [https://code.visualstudio.com/#alt-downloads](https://code.visualstudio.com/#alt-downloads) and download it and copy to `/vscode/ -->
5. Reload the dev container window.