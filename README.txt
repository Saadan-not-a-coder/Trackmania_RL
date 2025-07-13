🏎️ Trackmania United Forever AI Controller
This project contains Python code to control a car in Trackmania United Forever using a basic AI agent. The car runs on a custom track, making decisions automatically and resetting if it gets stuck.

🎮 Game Setup & Requirements
Game: Trackmania United Forever

Track: A custom-designed track (should be loaded and launched manually)

Display: Must be in fullscreen mode

The resolution can be adjusted in the code to match your screen size.

⚙️ How to Run
Open the Python .py script in your preferred editor or terminal.

Launch Trackmania United Forever and load your custom track.

When the car is placed at the starting "GO" position, proceed to the next step.

Run the Python code.

Quickly switch to the game window (Alt + Tab if needed).

The AI will begin controlling the car automatically!

⌨️ Controls
Key	Action
W	Move forward
A	Steer left
S	Brake / Reverse
D	Steer right
Enter	Reset car to start (manual)

🔁 The AI will also automatically reset the car to the starting position if it detects the car is stuck.

🧠 How It Works
The AI captures the screen in real-time and uses basic decision-making to control the car.

Movement is handled through simulated keyboard input (w, a, s, d).

A stuck detection mechanism ensures the car doesn't get stuck indefinitely by automatically triggering a reset.

🛠️ Customization
Screen Resolution: Update the capture dimensions in the script to match your display if needed.

Track Layout: You can train or test the AI on any custom-built track, ideally designed with forks, dead ends, or decision points.

📌 Notes
Make sure no other window is interfering with the game during AI control.

Performance may vary depending on system specs and frame rate.