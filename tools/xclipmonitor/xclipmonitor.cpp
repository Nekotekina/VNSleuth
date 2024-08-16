#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/extensions/XInput.h>
#include <X11/extensions/XInput2.h>
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
#include <termios.h>
#include <unistd.h>

using namespace std::literals;

volatile bool g_stop = false;

std::string xsel(bool selection = false, bool clear = false)
{
	// -p: retrieve PRIMARY selection
	// -b: retrieve CLIPBOARD
	// -o: write to STDOUT
	// -c: clear
	// -t: timeout 100 ms
	std::string cmd = "-";
	cmd += selection ? "po" : "bo";
	cmd += clear ? "c" : "";
	const char* args[] = {"xsel", cmd.c_str(), "-t", "100", nullptr};

	int opipe_fd[2]{}; // [0] Reading from xsel stdout
	if (pipe2(opipe_fd, O_NONBLOCK) != 0) {
		perror("Failed to create pipe");
		std::exit(1);
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
		perror("execvp failed for 'xsel'");
		std::exit(1);
	}

	// Wait for child
	const auto s0 = std::chrono::steady_clock::now();
	while (true) {
		// Sleep for 5 ms, I guess there's no point in more complex waiting implementation
		usleep(5000);
		int wstatus = 0;
		if (waitpid(pid, &wstatus, WNOHANG) == pid) {
			if (int res = WEXITSTATUS(wstatus)) {
				perror("xsel failed");
				std::exit(res);
			}
			break;
		}
		const auto s1 = std::chrono::steady_clock::now();
		if (long d = std::chrono::duration_cast<std::chrono::milliseconds>(s1 - s0).count(); g_stop || d > 200) {
			// Kill xsel after 200 ms
			kill(pid, SIGKILL);
			waitpid(pid, nullptr, 0);
			pid = 0;
			break;
		}
	}

	std::string buf = "\0";
	if (pid != 0 && !g_stop) {
		// Read with size limit
		buf.resize(1000);
		buf.resize(std::max<ssize_t>(0, read(opipe_fd[0], buf.data(), buf.size())));
		if (buf.size() >= 1000)
			buf.clear();
	}

	close(opipe_fd[0]);
	close(opipe_fd[1]);
	return buf;
}

// Replace newlines with \t, remove trailing newlines also
std::string flatten(std::string buf)
{
	for (auto& ch : buf) {
		if (ch == '\n')
			ch = '\t';
	}
	while (buf.ends_with("\t"))
		buf.erase(buf.end() - 1);
	return buf;
}

void sigh(int) { g_stop = true; }

int main(int argc, char* argv[])
{
	signal(SIGINT, sigh);

	// Set terminal to raw mode
	static struct termios told {};
	if (tcgetattr(0, &told) < 0) {
		perror("tcgetattr failed");
		return 1;
	}

	told.c_lflag &= ~ICANON;
	told.c_lflag &= ~ECHO;
	told.c_cc[VMIN] = 1;
	told.c_cc[VTIME] = 0;
	if (tcsetattr(0, TCSANOW, &told) < 0) {
		perror("tcsetattr failed");
		return 1;
	}

	told.c_lflag |= ICANON;
	told.c_lflag |= ECHO;
	atexit([] {
		// Reset terminal mode
		tcsetattr(0, TCSADRAIN, &told);
	});

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

	XIEventMask mask[2];
	mask[0].deviceid = XIAllDevices;
	mask[0].mask_len = XIMaskLen(XI_LASTEVENT);
	mask[0].mask = (unsigned char*)calloc(mask[0].mask_len, sizeof(char));
	mask[1].deviceid = XIAllMasterDevices;
	mask[1].mask_len = XIMaskLen(XI_LASTEVENT);
	mask[1].mask = (unsigned char*)calloc(mask[1].mask_len, sizeof(char));
	XISetMask(mask[1].mask, XI_RawButtonPress);
	//XISetMask(mask[1].mask, XI_RawButtonRelease);

	XISelectEvents(disp, root, &mask[0], 2);
	free(mask[0].mask);
	free(mask[1].mask);

	// Workaround for "double clicks" (reason uncertain)
	auto last_rewrite = std::chrono::steady_clock::now();
	std::string buf, last_text;
	while (!g_stop) {
		// Wait for events with 300ms timeout
		struct pollfd pfd = {
			.fd = ConnectionNumber(disp),
			.events = POLLIN,
			.revents = 0,
		};

		struct pollfd ifd = {
			.fd = STDIN_FILENO,
			.events = POLLIN,
			.revents = 0,
		};

		struct pollfd pfds[2]{pfd, ifd}; // A copy of pfds
		if (XPending(disp) > 0 || (poll(pfds, sizeof(pfds) / sizeof(pfds[0]), 300), poll(&pfd, 1, 0) > 0)) {
			const auto s0 = std::chrono::steady_clock::now();
			XNextEvent(disp, &evt);
			const auto s1 = std::chrono::steady_clock::now();
			if (long d = std::chrono::duration_cast<std::chrono::milliseconds>(s1 - s0).count(); d > 100) {
				fprintf(stderr, "xclipmonitor: XNextEvent took too long: %ldms\n", d);
			}

			XGenericEventCookie* cookie = (XGenericEventCookie*)&evt.xcookie;
			if (XGetEventData(disp, cookie) && cookie->type == GenericEvent) {
				XIRawEvent* event = (XIRawEvent*)cookie->data;
				if (cookie->evtype == XI_RawButtonPress && event->detail == 9) {
					// Send ^A rewrite request
					// Retrieve current text selection and send it
					auto rwtime = std::chrono::steady_clock::now();
					if (rwtime - last_rewrite < 1s) {
						continue;
					} else if (auto sel = xsel(true, true); !sel.empty()) {
						putchar('\01');
						puts(flatten(sel).c_str());
					} else {
						putchar('\n');
					}
					last_rewrite = rwtime;
					fflush(stdout);
					continue;
				}
			}
		}

		// Send keypresses read from stdin as control characters
		if (poll(&ifd, 1, 0) > 0) {
			char c{};
			if (read(STDIN_FILENO, &c, 1) == 1) {
				// Replace "dangerous" Reload command with safer Edit
				if (c == 'r')
					c = 'e';
				if (c >= 'a' && c <= 'z') {
					// Convert lowercase letters
					c -= 'a';
					c += 1;
				}
				if (c == '\05' && argc > 1) {
					// Send ^S save request (unreliable)
					putchar('\x13');
					putchar('\n');
					fflush(stdout);
					usleep(500'000);
					// Execute bash script ("edit" request ^E)
					pid_t pid = fork();
					if (pid == 0) {
						// Execute child process
						execvp(argv[1], const_cast<char**>(argv + 1));
						perror("execvp failed for argv[1]");
						return 1;
					}
					waitpid(pid, nullptr, 0);
					c = '\x12'; // ^R - reload request
				}
				if (c > 0 && c < 32) {
					if (c == '\01') {
						// Send ^A rewrite request
						// Retrieve current text selection and send it
						auto rwtime = std::chrono::steady_clock::now();
						if (rwtime - last_rewrite < 1s) {
							c = '\n';
						} else if (auto sel = xsel(true); !sel.empty()) {
							last_rewrite = rwtime;
							putchar('\01');
							puts(flatten(sel).c_str());
							c = '\n';
						} else {
							last_rewrite = rwtime;
							// Single \n will be sent
						}
					} else {
						putchar(c);
					}
					if (c != '\n')
						putchar('\n');
					fflush(stdout);
				}
				if (c == '\03') {
					// Stop on Ctrl+C
					usleep(300'000);
					break;
				}
			} else {
				perror("Reading from stdin failed");
				return 1;
			}

			// Retry poll for non-blocking read
			continue;
		}

		std::string buf = xsel(false);
		if (!buf.empty() && buf != last_text) {
			// Filter duplications
			last_text = buf;
			puts(flatten(buf).c_str());
			fflush(stdout);
		} else if (buf.empty()) {
			last_text.clear();
		}
	}

	XCloseDisplay(disp);
	return 0;
}
