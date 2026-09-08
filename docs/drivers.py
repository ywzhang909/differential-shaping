from typing import Tuple

import time
import logging
import socket

import numpy as np

import gxipy as gx

import utils


class CameraStreamManager:
    def __init__(self, cam_id: int = 0, framerate: float = 20, explosure_time: float = 0, background=0,
                 skip_sampling=False,
                 log=logging.getLogger('galaxy camera driver')):
        self.device_manager = gx.DeviceManager()
        self.cam_id = cam_id
        self.framerate = framerate
        self.explore_time = explosure_time
        self.background = background
        self.skip_sampling = skip_sampling

        self.cam, self.__sn = None, None
        self.cam_width, self.cam_height = 0, 0
        self.log = log

        self.frames = 0  # 当前帧数
        self.start_time = None  # 开始时间
        self.current_fps = "counting fps..."  # 当前帧率字符串
        self.htm_fps = 1  # 全局变量，用于存储当前帧率

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cam:
            self.cam_width, self.cam_height = 0, 0
            self.cam.stream_off()
            self.cam.close_device()
            self.cam, self.__sn = None, None

    def initialize(self):
        """
        初始化相机设备。

        此方法执行以下操作：
        1. 关闭之前打开的相机设备（如果有）。
        2. 更新设备列表并检查是否有足够的设备。
        3. 打开指定的相机设备。
        4. 设置相机的曝光时间、增益、像素格式、采样方式、偏移量、宽度和高度。
        5. 更新相机的属性并开启数据流。

        如果没有找到相机设备，将记录错误并抛出连接中止错误。

        参数:
            无

        返回:
            无
        """
        # 关闭之前打开的相机设备（如果有）
        self.__exit__(None, None, None)

        # 更新设备列表并获取设备信息列表
        _, dev_info_list = self.device_manager.update_device_list()
        # 检查设备列表长度是否小于等于指定的相机ID
        if len(dev_info_list) <= self.cam_id:
            self.log.error("No devices found.")
            raise ConnectionAbortedError("No cam devices found.")

        sn = dev_info_list[self.cam_id].get("sn")
        self.cam = self.device_manager.open_device_by_sn(sn)

        # 设置采集帧频
        self.cam.AcquisitionFrameRateMode.set(1)
        self.cam.AcquisitionFrameRate.set(self.framerate)

        # 设置相机的曝光时间
        self.cam.ExposureTime.set(self.explore_time)
        # 设置相机的增益
        self.cam.Gain.set(0.0)
        # 设置相机的像素格式为MONO8
        self.cam.PixelFormat.set(gx.GxPixelFormatEntry.MONO8)
        if self.skip_sampling:
            # 设置相机的合并因子为2
            self.cam.BinningHorizontal.set(2)
            self.cam.BinningVertical.set(2)

        # 设置相机的水平偏移量为0
        self.cam.OffsetX.set(0)
        self.cam.OffsetY.set(0)
        # 设置相机的宽度为最大宽度
        self.cam.Width.set(self.cam.WidthMax.get())
        self.cam.Height.set(self.cam.HeightMax.get())

        self.__sn = sn
        self.__update_properties()
        self.cam.stream_on()

    def reset_explore_time(self, time: int):
        if time >= 20:
            self.explore_time = time
        else:
            self.explore_time = 2000
            self.log.warning('explore time must >= 20. set to 20.')
        self.cam.ExposureTime.set(self.explore_time)
        return self.explore_time

    def reset_window(self, size: Tuple[int], center: Tuple[int]) -> Tuple[int]:
        """
        重置相机的窗口大小和位置，以确保图像的中心位于指定的位置。

        参数:
        size (Tuple[int]): 期望的窗口大小，格式为 (宽度, 高度)。
        center (Tuple[int]): 期望的窗口中心位置，格式为 (x坐标, y坐标)。

        返回:
        Tuple[int]: 新的窗口中心位置，格式为 (x坐标, y坐标)。
        """
        # 中心坐标大于0
        assert center[0] > 0 and center[1] > 0
        if self.cam:
            self.cam.stream_off()
            # 如果未指定窗口大小，则使用相机的最大宽度和高度
            if size == None:
                size = (self.cam.WidthMax.get(), self.cam.HeightMax.get())
            width, height = size
            width, height = width // 4 * 4, height // 4 * 4
            # 计算窗口的偏移量，确保中心位置在指定位置
            x_offset, y_offset = center[0] - (width // 2), center[1] - (height // 2)
            assert x_offset > 0 and y_offset > 0
            self.cam.Width.set(width)
            self.cam.Height.set(height)
            self.cam.OffsetX.set(x_offset // 4 * 4)
            # 设置相机的垂直偏移量，确保偏移量是4的倍数
            self.cam.OffsetY.set(y_offset // 4 * 4)

            self.__update_properties()
            self.cam.stream_on()

            # 返回新的窗口中心位置
            return (width, height), (width // 2, height // 2)

    def get_numpy_image(self) -> np.ndarray:
        background = self.background
        while True:
            # 计算相机采集帧率
            # if self.start_time is None or time.time() - self.start_time > 1.0:  # 每隔 1 秒更新一次
            #     elapsed_time = max(1.0, time.time() - self.start_time) if self.start_time else 1.0
            #     fps = self.frames / elapsed_time  # 计算帧率
            #     # 处理无效值
            #     if fps != fps or fps == float('inf'):  # 检查 NaN 或 Inf
            #         fps = 1
            #     # 更新帧率字符串
            #     if fps == 0:
            #         self.current_fps = "counting fps..."
            #     else:
            #         self.current_fps = f"{int(round(fps))} FPS"
            #     # 更新全局变量
            #     self.htm_fps = fps
            #     # 重置计时器和帧数
            #     self.start_time = time.time()
            #     self.frames = 0
            #     # 日志输出（调试用）
            #     print(f"Current FPS: {self.current_fps}")
            # else:
            #     self.frames += 1  # 增加帧计数

            raw_image = self.cam.data_stream[0].get_image()
            if not raw_image:
                continue
            numpy_image = raw_image.get_numpy_array()
            numpy_image = numpy_image.astype(np.int16)
            correct_numpy_image = numpy_image - background
            correct_numpy_image = np.maximum(correct_numpy_image, 0)
            correct_numpy_image = correct_numpy_image.astype(np.uint8)
            return correct_numpy_image.astype(int)

    def __update_properties(self):
        self.cam_width = self.cam.Width.get()
        self.cam_height = self.cam.Height.get()
        self.log.info(f"Open cam {self.__sn} success. width={self.cam_width}, height={self.cam_height}")
        self.xv, self.yv = self.__get_grid(self.cam_width, self.cam_height)

    @staticmethod
    def __get_grid(width, height):
        x = np.arange(0, width)
        y = np.arange(0, height)
        xv, yv = np.meshgrid(x, y)
        return xv, yv


DM_IP = "192.168.6.10"
DM_PORT = 1001
HEAD_WITH_ECHO = '10 01 2c'.split(' ')
HEAD = '30 01 2d'.split(' ')
mirror_nums = 64
REG_IDS = [0, 16384, 32768, 49152]  # 0x00, 0x40, 0x80, 0xc0 to dec


class DmUDP:
    def __init__(self):
        self.ip = DM_IP
        self.port = DM_PORT
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.dm_num = mirror_nums

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.sock:
            self.sock.close()
            self.sock = None

    @staticmethod
    def _num_hex(num: int):
        hex_16 = hex(int(num))[2:].zfill(4)
        return hex_16[:2] + ' ' + hex_16[2:]

    @staticmethod
    def _voltage_hex(num: float, registry: int):
        _num = int((num + 500) / 1000 * 4096)
        _num = min(_num, 4095)
        _num = max(_num, 820)
        _num += REG_IDS[registry % 4]
        hex_16 = DmUDP._num_hex(_num)
        return hex_16

    def initialize(self) -> None:
        self.set_hv()
        self.reset_all()

    def send(self, message):
        hex_message = bytes.fromhex(message)
        return self.sock.sendto(hex_message, (self.ip, self.port))

    def reset_all(self):
        vs = np.zeros(256)
        send_data = ' '.join(
            HEAD_WITH_ECHO + [self._num_hex(256)] + [self._voltage_hex(v, i + 1) for i, v in enumerate(vs)])
        ret = self.send(send_data) & self.send("10 00 00 00 01 00 03")

        time.sleep(0.5)
        return ret

    def send_voltages(self, vs: np.ndarray, wait_time_s=0.001):
        # vs = vs.astype(np.int16)
        vs = np.clip(vs, -299, 499)  # 限制电压范围在-300到499之间
        send_data = ' '.join(
            HEAD + [self._num_hex(mirror_nums)] + [self._voltage_hex(v, i + 1) for i, v in enumerate(vs)])
        ret = self.send(send_data)
        time.sleep(wait_time_s)
        return ret

    def set_hv(self):
        code = '03 00 03 00 01'
        ret = self.send(code)
        # time.sleep(0.5)
        return ret


try:
    import bmc


    class BMCManager:

        def __init__(self, sn='17DW023#013', log=logging.getLogger('bmc dm driver')):

            self.sn = sn
            self.dm: bmc.BmcDm = None
            self.dm_num = 140
            self.log = log

        def initialize(self):
            self.dm = bmc.BmcDm()
            res = self.dm.open_dm(self.sn)
            if res:
                self.log.error(self.dm.error_string(res))
                return

            self.dm_num = self.dm.num_actuators()

            self.log.info(f'dm {self.sn} init. actuators count {self.dm_num}.')

        def __enter__(self):
            self.initialize()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self.dm:
                self.dm.close_dm()
            self.dm = None

        def send_voltages(self, data):
            res = self.dm.send_data(data)
            if res:
                self.log.error(self.dm.error_string(res))

    class BMCManager2:

        def __init__(self, sn='17DW022#037', log=logging.getLogger('bmc dm driver')):

            self.sn = sn
            self.dm: bmc.BmcDm = None
            self.dm_num = 140
            self.log = log

        def initialize(self):
            self.dm = bmc.BmcDm()
            res = self.dm.open_dm(self.sn)
            if res:
                self.log.error(self.dm.error_string(res))
                return

            self.dm_num = self.dm.num_actuators()

            self.log.info(f'dm {self.sn} init. actuators count {self.dm_num}.')

        def __enter__(self):
            self.initialize()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            if self.dm:
                self.dm.close_dm()
            self.dm = None

        def send_voltages(self, data):
            res = self.dm.send_data(data)
            if res:
                self.log.error(self.dm.error_string(res))
except Exception as e:
    print('use python 3.6 for bmc dm.')

if __name__ == '__main__':
    import numpy as np

    with DmUDP() as dm:
        # v = np.random.uniform(-300, 499, dm.dm_num)
        v = np.zeros(dm.dm_num)
        v[0] = 0
        vs = v.astype(np.int16)
        vs = np.clip(vs, -300, 499)
        send_data = ' '.join(HEAD + [dm._num_hex(mirror_nums)] + [dm._voltage_hex(v, i + 1) for i, v in enumerate(vs)])
        print(send_data)
