import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import socket
import sys

# Create figure for plotting
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
xs = []
ys1 = []
ys2 = []
y_exercise = []
counter = 0

# Set server
ip = '127.0.0.1'
port = 4444

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP
server_address = (ip, port)
sock.bind(server_address)
print('Connected and waiting to receive...')

# This function returns an int corresponding to the received data via UDP
def receive_udp(port):
    data, addr = sock.recvfrom(port)  # buffer size is 1024 bytes
    data = data.decode("utf-8")
    data = data.replace('[','').replace(']','')
    data = data.split(", ")
    return data, addr

# This function is called periodically from FuncAnimation
def animate(i, xs, ys1, ys2, y_exercise):
    global counter
    counter += 1
    # Call the function to receive the data via UDP protocol
    data, addr = receive_udp(1024)
    print("Message received from main: {}".format(data))

    if int(data[0])==1 and int(data[1]) == 100:
        sock.close()
    else:
        angle_knee_xy = int(data[0])
        angle_knee_xyz = int(data[1])
        data_exercise = int(data[2])

        # Add x and y to lists
        xs.append(counter)
        ys1.append(angle_knee_xy)
        ys2.append(angle_knee_xyz)
        y_exercise.append(data_exercise)

        # Limit x and y lists to 20 items
        xs = xs[-500:]
        ys1 = ys1[-500:]
        ys2 = ys2[-500:]
        y_exercise = y_exercise[-500:]

        # Draw x and y lists in the ax1 plot (knee)
        ax1.clear()
        ax1.plot(xs, ys1, label='Sagital')
        ax1.plot(xs, ys2, label='3D')
        ax1.plot(xs, y_exercise, label='Proposed Degree')
        ax1.set_xlabel('time')
        ax1.set_ylabel('Knee Degree (C)')
        ax1.legend(['Sagital','3D','Proposed Degree'])

        # Format plot
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.30)
        plt.title('Angles over Time')
        plt.ylabel('degree (C)')

        # Send back a signal to continue the execution
        send_data = "Angle plotted, continue..."
        sock.sendto(send_data.encode("utf-8"), addr)
        print("Message sent to main: {}".format(send_data.encode("utf-8")))

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, fargs=(xs, ys1, ys2, y_exercise), interval=1)
plt.show()
