#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/extensions/Xfixes.h>
#include <algorithm>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

volatile bool g_stop = false;

void sigh(int) { g_stop = true; }

int main()
{
	signal(SIGINT, sigh);

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

	// -b: retrieve CLIPBOARD
	// -t: timeout 100 ms
	const char* args[] = {"xsel", "-b", "-t", "100", nullptr};

	std::string buf, last_text;
	while (!g_stop) {
		// Wait for events with 300ms timeout
		struct pollfd pfd = {
			.fd = ConnectionNumber(disp),
			.events = POLLIN,
			.revents = 0,
		};

		if (XPending(disp) > 0 || poll(&pfd, 1, 300) > 0) {
			const auto s0 = std::chrono::steady_clock::now();
			XNextEvent(disp, &evt);
			const auto s1 = std::chrono::steady_clock::now();
			if (long d = std::chrono::duration_cast<std::chrono::milliseconds>(s1 - s0).count(); d > 100) {
				fprintf(stderr, "xclipmonitor: XNextEvent took too long: %ldms\n", d);
			}
		}

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
			perror("execvp failed for 'xsel':");
			return 1;
		}

		// Wait for child
		{
			const auto s0 = std::chrono::steady_clock::now();
			while (true) {
				// Sleep for 5 ms, I guess there's no point in more complex waiting implementation
				usleep(5000);
				int wstatus = 0;
				if (waitpid(pid, &wstatus, WNOHANG) == pid) {
					if (int res = WEXITSTATUS(wstatus)) {
						perror("xsel failed:");
						return res;
					}
					break;
				}
				const auto s1 = std::chrono::steady_clock::now();
				if (long d = std::chrono::duration_cast<std::chrono::milliseconds>(s1 - s0).count(); d > 200) {
					// Kill xsel after 200 ms
					kill(pid, SIGKILL);
					waitpid(pid, nullptr, 0);
					pid = 0;
					break;
				}
			}

			if (pid == 0) {
				close(opipe_fd[0]);
				close(opipe_fd[1]);
				continue;
			}
		}

		// Read with size limit
		buf.resize(1000);
		buf.resize(std::max<ssize_t>(0, read(opipe_fd[0], buf.data(), buf.size())));
		if (buf.size() >= 1000)
			buf.clear();
		if (!buf.empty() && buf != last_text) {
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
		} else if (buf.empty()) {
			last_text.clear();
		}

		close(opipe_fd[0]);
		close(opipe_fd[1]);
	}

	XCloseDisplay(disp);
	return 0;
}
