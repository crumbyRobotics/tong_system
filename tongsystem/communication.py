import zmq
import numpy as np
import pickle


class SocketManager:
    def __init__(self, ip_addr: str = "*:5555"):
        ctx = zmq.Context()
        self.socket = ctx.socket(zmq.REP)
        self.socket.bind(f"tcp://{ip_addr}")

        self.sim_protocol = False  # True when tongsim # NOTE Automatically detected when receive

    def receive(self) -> tuple[tuple, dict[dict, np.ndarray]]:
        recv = self.socket.recv_multipart()

        if not self.sim_protocol and len(recv) > 14:
            # NOTE Once sim_protocol is flagged, it cannot be changed later.
            self.sim_protocol = True

        # Recieve robot state
        posN = np.frombuffer(recv[0], dtype=np.float64).copy()
        posL = np.frombuffer(recv[1], dtype=np.float64).copy()
        posR = np.frombuffer(recv[2], dtype=np.float64).copy()
        velN = np.frombuffer(recv[3], dtype=np.float64).copy()
        velL = np.frombuffer(recv[4], dtype=np.float64).copy()
        velR = np.frombuffer(recv[5], dtype=np.float64).copy()
        fsL = np.frombuffer(recv[6], dtype=np.float64).copy()
        fsR = np.frombuffer(recv[7], dtype=np.float64).copy()
        LInfo = np.frombuffer(recv[8], dtype=np.int32)
        Lrows, Lcols, Lchannels = LInfo[0], LInfo[1], LInfo[2]
        SbSResultL = np.frombuffer(recv[9], dtype=np.uint8).reshape(Lrows, Lcols, Lchannels).copy()
        RInfo = np.frombuffer(recv[10], dtype=np.int32)
        Rrows, Rcols, Rchannels = RInfo[0], RInfo[1], RInfo[2]
        SbSResultR = np.frombuffer(recv[11], dtype=np.uint8).reshape(Rrows, Rcols, Rchannels).copy()
        DInfo = np.frombuffer(recv[12], dtype=np.int32)
        Drows, Dcols, Dchannels = DInfo[0], DInfo[1], DInfo[2]
        SbSResultD = np.frombuffer(recv[13], dtype=np.uint16).reshape(Drows, Dcols, Dchannels).copy()

        robot_state = (posN, posL, posR, velN, velL, velR, fsL, fsR, SbSResultL, SbSResultR, SbSResultD)

        # Recieve world state
        world_state = {}
        if self.sim_protocol:
            world_state = pickle.loads(recv[14])

        return robot_state, world_state

    def send(self, cmd_params: np.ndarray, cmdN: np.ndarray, cmdL: np.ndarray, cmdR: np.ndarray, reset: bool):
        self.socket.send(cmd_params, zmq.SNDMORE)
        self.socket.send(cmdN.astype(np.float64), zmq.SNDMORE)
        self.socket.send(cmdL.astype(np.float64), zmq.SNDMORE)
        if self.sim_protocol:
            self.socket.send(cmdR.astype(np.float64), zmq.SNDMORE)
            self.socket.send(b"\x01" if reset else b"\x00")
        else:
            self.socket.send(cmdR.astype(np.float64))
