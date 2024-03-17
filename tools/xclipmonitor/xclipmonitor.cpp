#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/extensions/Xfixes.h>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main()
{
	Display* disp;
	Window root;
	Atom clip;
	XEvent evt;

	disp = XOpenDisplay(nullptr);
	if (!disp) {
		perror("XOpenDisplay failed:");
		return 1;
	}

	root = DefaultRootWindow(disp);
	clip = XInternAtom(disp, "CLIPBOARD", False);
	XFixesSelectSelectionInput(disp, root, clip, XFixesSetSelectionOwnerNotifyMask);

	const char* args[] = {"xsel", "-b", nullptr};

	std::string buf, last_text;
	while (true) {
		XNextEvent(disp, &evt);

		int opipe_fd[2]{}; // [0] Reading from xsel stdout
		if (pipe2(opipe_fd, O_NONBLOCK) != 0) {
			perror("Failed to create pipe:");
			return 1;
		}

		// Fork a child process
		pid_t pid = fork();
		if (pid == 0) {
			// Inside the child process
			while ((dup2(opipe_fd[1], STDOUT_FILENO) == -1) && (errno == EINTR)) {}
			close(opipe_fd[0]);
			close(opipe_fd[1]);

			// Execute xsel
			execvp(args[0], const_cast<char**>(args));
			perror("execvp failed for 'xsel -b':");
			return 1;
		}

		// Wait for child
		int wstatus = 0;
		waitpid(pid, &wstatus, 0);
		if (int res = WEXITSTATUS(wstatus)) {
			perror("xsel failed:");
			return res;
		}

		// Read with size limit
		buf.resize(1000);
		buf.resize(std::max<ssize_t>(0, read(opipe_fd[0], buf.data(), buf.size())));
		if (!buf.empty() && buf.size() < 1000 && buf != last_text) {
			// Filter duplications
			last_text = buf;
			for (auto& ch : buf) {
				// Replace newlines with \t
				if (ch == '\n')
					ch = '\t';
			}
			while (buf.ends_with("\t"))
				buf.erase(buf.end() - 1);
			printf("%s\n", buf.c_str());
			fflush(stdout);
		} else {
			last_text.clear();
		}

		close(opipe_fd[0]);
		close(opipe_fd[1]);
	}

	// unreachable
	XCloseDisplay(disp);
	return 0;
}