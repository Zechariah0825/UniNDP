import numpy as np
import math

from CorrectLPDDR4_verify import n_tile

###

USE_LOOP = False
SHOW_DESinfo = False
# FORCE_LPDDR_FLOW = False

Channels_Perdie_HBM2 = 2 # 2 for HBM-PIM
PIMUnits_pBG_HBM2 = 2 # 2 for HBM-PIM
Chip_dies_HBM2 = 4 # 4 for HBM-PIM

Channels_Perdie_HBM3 = 2  # 2 for real-Attacc
PIMUnits_pBG_HBM3 = 4  # 4 for real-Attacc
Chip_dies_HBM3 = 8  # 8 for real-Attacc #todo{8 dies??? each with 4x2 pCH?}

Channels_Perdie_LPDDR4 = 1  # 1 for McDRAM
PIMUnits_pBANK_LPDDR4 = 1
Chip_dies_LPDDR4 = 8  # 8 for McDRAM


class PAX_unit:
    def __init__(self, PIM_type='HBM2',
                 # Channels_Perdie = 2,
                 # Chip_dies = 4,
                 # PIMUnits_pBG = 2,
                 # SIMDlane_num=16,
                 inBuf_size=8 * 16 * 16, outBuf_size=8 * 16 * 16, Run_Baseline=False,
                 PAX_Num=4):
        self.PIM_type = PIM_type
        self.Run_Baseline = Run_Baseline
        self.PAX_Num = PAX_Num

        # HBM2
        if self.PIM_type == 'HBM2':
            self.ACT_WAY = 'GACT'
            # timing parameters
            self.CLK_freq = 1.2 * 1e9
            self.tCK = 1

            self.tRRDS = 4 * self.tCK
            self.tRRDL = 6 * self.tCK
            self.tCCDS = 2 * self.tCK
            self.tCCDL = 4 * self.tCK
            self.tCCDR = 3 * self.tCK
            self.tRCDRD = 14 * self.tCK
            self.tRCDWR = 10 * self.tCK
            self.tRTPS = 4 * self.tCK
            self.tRTPL = 5 * self.tCK
            self.WL = 8 * self.tCK
            self.AL = 0 * self.tCK
            self.RL = 20 * self.tCK
            self.BL = 4
            self.BL_time = self.BL * self.tCK
            self.tWR = 16 * self.tCK
            self.tFAW = 16 * self.tCK
            self.tRAS = 33 * self.tCK
            self.tRP = 14 * self.tCK
            self.tRC = 47 * self.tCK  # (= tRAS + tRP

            # self.WRITE_TO_PRE_lim = max((self.tRAS - self.tRCDWR), (self.WL + self.BL_time / 2 + self.tWR))
            # self.Extra_WRITE_WAIT_lim = (self.WL + self.BL_time / 2 + self.tWR)
            # self.RD_TO_PRE_lim = max((self.tRAS - self.tRCDRD), self.tRTPL)

            self.WRITE_TO_PRE_lim = max((self.tRAS - self.tRCDWR), (self.WL + self.BL_time / 2 + self.tWR))
            self.Extra_WRITE_WAIT_lim = (self.WL + self.BL_time / 2 + self.tWR)
            self.RD_TO_PRE_lim = self.tRAS - self.tRCDRD
            self.PRE_AFT_lim = self.tRP

            # PIM parameters
            # self.Channel_Bitwidth = 64
            # self.DRAM_Bandwidth = 64  # todo{TO verify it!!}
            self.DQ_bitwidth = 64
            self.Burst_len = self.BL * self.DQ_bitwidth
            self.Bank_Row_size = 32 * self.DQ_bitwidth * self.BL

            self.Channels_Perdie = Channels_Perdie_HBM2
            self.Chip_dies = Chip_dies_HBM2

            self.Channel_num = self.Chip_dies * self.Channels_Perdie
            self.pChannel_num = 2 * self.Channel_num  # pCH is Vital
            if self.Channel_num > 8:
                print("Too More Channels!")
                exit()

            self.Bank_num_pBG = 4
            self.BG_num_ppCH = 4
            self.Bank_num = self.Bank_num_pBG * self.BG_num_ppCH * self.pChannel_num

            self.PIMUnits_pBG = PIMUnits_pBG_HBM2
            self.PIMUnits_PerBank = self.PIMUnits_pBG / self.Bank_num_pBG

            self.PE_num = int(self.Bank_num * self.PIMUnits_PerBank)  # PE_num is Vital
            if not (self.PE_num / self.Bank_num == self.PIMUnits_PerBank):
                print("Wrong PE_num / Bank_num")
                exit()
            self.ResNumPerUnit = 1
            if Run_Baseline:
                self.ResNumPerUnit = 16
                print("Set HBM2-PIM save {} OPs per Unit".format(self.ResNumPerUnit))

            # self.PIM_num_perpCh = int((self.Bank_num_pBG * self.BG_num_ppCH) * self.PIMUnits_PerBank)
            self.PIM_num_perChannel = int((self.Bank_num_pBG * self.BG_num_ppCH * 2 * 1) * self.PIMUnits_PerBank)

            # self.SIMDlane_num = SIMDlane_num


            self.localbuf_input = inBuf_size  # PIMSimu: 8 * 16 FP16 = 128 FP16     should be 2^n * FP16 bit !!!!!
            self.localbuf_output = outBuf_size  # PIMSimu: 8 * 16 FP16 = 128 FP16    should be 2^n * FP16 bit !!!!!
            self.LocalBuffer_size = self.localbuf_input + self.localbuf_output

            # ACT parameters
            if self.ACT_WAY == 'GACT':
                self.ACT_TO_WRITE_lim = self.tRCDWR
                self.ACT_TO_RD_lim = self.tRCDRD
            elif self.ACT_WAY == 'pGACT':
                self.ACT_TO_WRITE_lim = math.ceil(self.BG_num_ppCH / 4) * (self.PIMUnits_pBG) * self.tFAW + int(
                    self.tFAW < self.tRCDWR) * (self.tRCDWR - self.tFAW)
                self.ACT_TO_RD_lim = math.ceil(self.BG_num_ppCH / 4) * (self.PIMUnits_pBG) * self.tFAW + int(
                    self.tFAW < self.tRCDRD) * (self.tRCDRD - self.tFAW)
            elif self.ACT_WAY == 'NORM':
                self.ACT_TO_WRITE_lim = self.tRCDWR
                self.ACT_TO_RD_lim = self.tRCDRD
            else:
                print("WRONG ACT_WAY: {}".format(self.ACT_WAY))

            # self.CompPwr_origin = self.PE_num * self.SIMDlane_num * self.CLK_freq / 1e9 #todo{Should Verify it!!!}
            # self.BW_origin = self.PE_num * self.Burst_len * 2 * (self.CLK_freq / self.BL) / 1e9 #todo{Should Verify it!!!}
        elif self.PIM_type == 'HBM3':
            # timing parameters
            self.CLK_freq = 1.3e9
            # self.tCK = 1 / self.CLK_freq
            self.tCK = 1

            self.tRRDS = 2 * self.tCK
            self.tRRDL = 4 * self.tCK
            self.tCCDS = 2 * self.tCK
            self.tCCDL = 4 * self.tCK
            # self.tCCDR = 3 * self.tCK
            self.tRCDRD = 19 * self.tCK
            self.tRCDWR = 19 * self.tCK
            self.tRTPS = 6 * self.tCK
            self.tRTPL = 8 * self.tCK
            self.WL = 6 * self.tCK
            # self.AL = 0 * self.tCK
            self.BL = 8  ############################## ???
            self.BL_time = self.BL * self.tCK
            self.tWR = 21 * self.tCK
            self.tFAW = 39 * self.tCK
            self.tRAS = 45 * self.tCK
            self.tRP = 19 * self.tCK
            self.tRC = 63 * self.tCK  # (= tRAS + tRP)
            self.WRITE_TO_PRE_lim = max((self.tRAS - self.tRCDWR), (self.WL + self.BL_time / 2 + self.tWR))
            self.Extra_WRITE_WAIT_lim = (self.WL + self.BL_time / 2 + self.tWR)
            self.RD_TO_PRE_lim = self.tRAS - self.tRCDRD
            self.PRE_AFT_lim = self.tRP

            # PIM parameters
            # self.Channel_Bitwidth = 64
            # self.DRAM_Bandwidth = 32  # todo{TO verify it!!}
            self.DQ_bitwidth = 32
            self.Burst_len = self.BL * self.DQ_bitwidth
            self.Bank_Row_size = 32 * self.DQ_bitwidth * self.BL

            self.Channels_Perdie = Channels_Perdie_HBM3
            self.Chip_dies = Chip_dies_HBM3

            self.Channel_num = self.Chip_dies * self.Channels_Perdie
            self.pChannel_num = 2 * self.Channel_num  # pCH is Vital
            if self.Channel_num > 16:
                print("Too More Channels!")
                exit()

            self.Bank_num_pBG = 4
            self.BG_num_ppCH = 4
            self.Bank_num = self.Bank_num_pBG * self.BG_num_ppCH * self.pChannel_num

            self.PIMUnits_pBG = PIMUnits_pBG_HBM3
            self.PIMUnits_PerBank = self.PIMUnits_pBG / self.Bank_num_pBG

            self.PE_num = int(self.Bank_num * self.PIMUnits_PerBank)  # PE_num is Vital
            if not (self.PE_num / self.Bank_num == self.PIMUnits_PerBank):
                print("Wrong PE_num / Bank_num")
                exit()
            self.ResNumPerUnit = 1


            self.PIM_num_perpCh = int((self.Bank_num_pBG * self.BG_num_ppCH) * self.PIMUnits_PerBank)
            self.PIM_num_perChannel = int((self.Bank_num_pBG * self.BG_num_ppCH * 2 * 1) * self.PIMUnits_PerBank)

            # self.OP_Bitwidth = OP_Bitwidth

            # self.SIMDlane_num = SIMDlane_num


            self.localbuf_input = inBuf_size  # PIMSimu: 8 * 16 FP16 = 128 FP16     should be 2^n * FP16 bit !!!!!
            self.localbuf_output = outBuf_size  # PIMSimu: 8 * 16 FP16 = 128 FP16    should be 2^n * FP16 bit !!!!!
            self.LocalBuffer_size = self.localbuf_input + self.localbuf_output

            # ACT parameters
            if self.ACT_WAY == 'GACT':
                self.ACT_TO_WRITE_lim = self.tRCDWR
                self.ACT_TO_RD_lim = self.tRCDRD
            elif self.ACT_WAY == 'pGACT':
                self.ACT_TO_WRITE_lim = math.ceil(self.BG_num_ppCH / 4) * (self.PIMUnits_pBG) * self.tFAW + int(
                    self.tFAW < self.tRCDWR) * (self.tRCDWR - self.tFAW)
                self.ACT_TO_RD_lim = math.ceil(self.BG_num_ppCH / 4) * (self.PIMUnits_pBG) * self.tFAW + int(
                    self.tFAW < self.tRCDRD) * (self.tRCDRD - self.tFAW)
            elif self.ACT_WAY == 'NORM':
                self.ACT_TO_WRITE_lim = self.tRCDWR
                self.ACT_TO_RD_lim = self.tRCDRD
            else:
                print("WRONG ACT_WAY: {}".format(self.ACT_WAY))

            # self.CompPwr_origin = self.PE_num * self.SIMDlane_num * self.CLK_freq / 1e9 #todo{Should Verify it!!!}
            # self.BW_origin = self.PE_num * self.Burst_len * 2 * (self.CLK_freq / self.BL) / 1e9 #todo{Should Verify it!!!}

        elif self.PIM_type == 'LPDDR4':
            # self.ACT_WAY = 'GACT'
            self.ACT_WAY = 'NORM'

            # timing parameters
            self.CLK_freq = 2 * 1e9  # not sure!
            self.tCK = 1
            self.tRRD = 15 * self.tCK
            self.tCCD = 8 * self.tCK
            self.tCCDL = self.tCCD  # duplicate for PAX model
            self.tRCD = 36 * self.tCK
            self.tRCDRD = self.tRCD
            self.tRCDWR = self.tRCD
            self.tRTP = 16 * self.tCK
            self.tRTPL = self.tRTP
            self.WL = 18 * self.tCK
            self.AL = 0 * self.tCK
            self.RL = 36 * self.tCK
            self.BL = 16
            self.BL_time = self.BL * self.tCK
            self.tWR = 40 * self.tCK
            self.tFAW = 60 * self.tCK
            self.tRAS = 84 * self.tCK
            self.tRP = 42 * self.tCK
            self.tRC = 126 * self.tCK  # (= tRAS + tRP
            self.WRITE_TO_PRE_lim = max((self.tRAS - self.tRCD), (self.WL + self.BL_time / 2 + self.tWR))  # by Cui
            self.Extra_WRITE_WAIT_lim = (self.WL + self.BL_time / 2 + self.tWR)
            self.RD_TO_PRE_lim = self.tRAS - self.tRCD  # by Cui
            self.PRE_AFT_lim = self.tRP  # by Cui

            # PIM parameters
            # self.Channel_Bitwidth = 64
            # self.DRAM_Bandwidth = 64  # todo{TO verify it!!}
            self.DQ_bitwidth = 16
            self.Burst_len = self.BL * self.DQ_bitwidth
            self.Bank_Row_size = 64 * self.DQ_bitwidth * self.BL

            self.Channels_Perdie = Channels_Perdie_LPDDR4
            self.Chip_dies = Chip_dies_LPDDR4

            self.Channel_num = self.Chip_dies * self.Channels_Perdie
            self.Bank_num_pCH = 8
            self.Bank_num = self.Bank_num_pCH * self.Channel_num

            self.PIMUnits_PerBank = PIMUnits_pBANK_LPDDR4

            self.PE_num = int(self.Bank_num * self.PIMUnits_PerBank)  # PE_num is Vital
            if not (self.PE_num / self.Bank_num == self.PIMUnits_PerBank):
                print("Wrong PE_num / Bank_num")
                exit()
            self.ResNumPerUnit = 1

            # self.PIM_num_perpCh = int((self.Bank_num_pBG * self.BG_num_ppCH) * self.PIMUnits_PerBank)
            self.PIM_num_perChannel = int(self.Bank_num_pCH * self.PIMUnits_PerBank)

            # self.OP_Bitwidth = OP_Bitwidth

            # self.SIMDlane_num = SIMDlane_num

            self.localbuf_input = inBuf_size  # PIMSimu: 8 * 16 FP16 = 128 FP16     should be 2^n * FP16 bit !!!!!
            self.localbuf_output = outBuf_size  # PIMSimu: 8 * 16 FP16 = 128 FP16    should be 2^n * FP16 bit !!!!!
            self.LocalBuffer_size = self.localbuf_input + self.localbuf_output

            # ACT parameters
            if self.ACT_WAY == 'GACT':
                self.ACT_TO_WRITE_lim = self.tRCDWR
                self.ACT_TO_RD_lim = self.tRCDRD
            elif self.ACT_WAY == 'pGACT':
                self.ACT_TO_WRITE_lim = math.ceil(self.BG_num_ppCH / 4) * (self.PIMUnits_pBG) * self.tFAW + int(
                    self.tFAW < self.tRCDWR) * (self.tRCDWR - self.tFAW)
                self.ACT_TO_RD_lim = math.ceil(self.BG_num_ppCH / 4) * (self.PIMUnits_pBG) * self.tFAW + int(
                    self.tFAW < self.tRCDRD) * (self.tRCDRD - self.tFAW)
            elif self.ACT_WAY == 'NORM':
                xFAW = math.floor(self.PIM_num_perChannel / 4)
                rem_banks = self.PIM_num_perChannel - xFAW * 4
                self.ACT_TO_WRITE_lim = xFAW * self.tFAW + rem_banks * self.tRRD - self.tRRD + self.tRCDWR
                self.ACT_TO_RD_lim = xFAW * self.tFAW + rem_banks * self.tRRD - self.tRRD + self.tRCDRD
            else:
                print("WRONG ACT_WAY: {}".format(self.ACT_WAY))

            # self.CompPwr_origin = self.PE_num * self.SIMDlane_num * self.CLK_freq / 1e9 #todo{Should Verify it!!!}
            # self.BW_origin = self.PE_num * self.Burst_len * 2 * (self.CLK_freq / self.BL) / 1e9 #todo{Should Verify it!!!}


        else:
            print("Wrong PIM_type {}".format(PIM_type))
            exit()
        assert self.ResNumPerUnit is not int, "ResNumPerUnit({}) is not int at {}-PIM".format(self.ResNumPerUnit, self.PIM_type)
        # self.VECbits = 0
        # self.MATbits = 0
        self.RESSAVE_INCLUDED = True
        self.ROWBUFMAX_ENABLEED = True



    def activate_VECtile(self, k_tile):
        repeat_num = math.ceil(k_tile * self.VECbits / self.Bank_Row_size)
        latency_vect_ACT = self.ACT_TO_RD_lim
        latency_vect_READ = math.ceil(min(k_tile * self.VECbits, self.Bank_Row_size) / self.Burst_len) * self.tCCDL
        latency_vect_PREC = self.PRE_AFT_lim
        # latency_vect = (latency_vect_ACT + max(latency_vect_READ, self.RD_TO_PRE_lim) + self.Extra_WRITE_WAIT_lim + latency_vect_PREC) * repeat_num  ## 86 cycles in trace!!!
        latency_vect = (latency_vect_ACT + max(
            (latency_vect_READ - self.tCCDL + max(self.RL + int(self.BL_time / 2), self.tRTPL)),
            self.RD_TO_PRE_lim) + latency_vect_PREC) * repeat_num  ## 86 cycles in trace!!!
        return latency_vect

    def MAC_subtile(self, k_subtile):
        bits_perMAC = min(self.Burst_len, self.MAC_Engine_size)
        # bits_perMAC = self.MAC_Engine_size

        total_repeat_num = math.ceil(k_subtile * self.MATbits / self.Bank_Row_size)
        latency_mat_ACT = self.ACT_TO_RD_lim
        latency_mat_mid = math.ceil(min(k_subtile * self.MATbits, self.Bank_Row_size) / bits_perMAC) * self.tCCDL
        latency_mat_PREC = self.PRE_AFT_lim
        # latency_mat = (latency_mat_ACT + max(latency_mat_mid, self.RD_TO_PRE_lim) + latency_mat_PREC) * total_repeat_num
        latency_mat = (latency_mat_ACT + max((latency_mat_mid - self.tCCDL + self.tRTPL),
                                             self.RD_TO_PRE_lim) + latency_mat_PREC) * total_repeat_num

        return latency_mat

    def result_save_subtile(self, ResNum_subtile):
        repeat_num = math.ceil(ResNum_subtile * self.VECbits / self.Bank_Row_size)

        latency_result_ACT = self.ACT_TO_WRITE_lim
        latency_result_WRITE = math.ceil(
            min(ResNum_subtile * self.VECbits, self.Bank_Row_size) / self.Burst_len) * self.tCCDL
        # latency_result_WRITE = math.ceil(8 * 16 * OP_Bitwidth/ Burst_len) * tCCDL  ###This is what PIMSimulator did
        latency_result_PREC = self.PRE_AFT_lim
        latency_result = (latency_result_ACT + max((latency_result_WRITE - self.tCCDL + self.Extra_WRITE_WAIT_lim),
                                                   self.WRITE_TO_PRE_lim) + latency_result_PREC) * repeat_num
        return latency_result


    def Build_Tile_DS(self):
        inBuf_gran = self.SIMDlane_num
        outBuf_gran = self.ResNumPerUnit # for HBM-PIM (or Accumulator), is equals to its #SIMDlane

        SubTile_k_MAX_gran = math.floor(self.SubTile_k_MAX / inBuf_gran)
        SubTile_n_MAX_gran = math.floor(self.SubTile_n_MAX / outBuf_gran)
        
        
        
        # max_PE = math.floor(math.log2(self.PE_num))
        max_PE = self.PE_num
        PEkPEn_list = []
        for i in range(max_PE):
            PEk = int(i + 1)

            PEn = math.floor(self.PE_num / PEk)
            assert PEk * PEn <= self.PE_num, "PEk PEn is larger!"
            PEkPEn_list.append([PEk, PEn])
        inoutBuf_gran_list = []
        for i in range(SubTile_k_MAX_gran):
            for j in range(SubTile_n_MAX_gran):
                inoutBuf_gran_list.append([i, j])

        PEkPEn_TkTn_inoutOP_list = []
        for PEkPEn in PEkPEn_list:
            PEk = PEkPEn[0]
            PEn = PEkPEn[1]
            for inoutBuf_granNum in inoutBuf_gran_list:
                inBuf_granNum = inoutBuf_granNum[0]
                outBuf_granNum = inoutBuf_granNum[1]
                inBuf_opNum = int(inBuf_granNum * inBuf_gran)
                outBuf_opNum = math.floor((outBuf_granNum * outBuf_gran) / self.ResNumPerUnit)

                Tk = int(PEk * inBuf_opNum)
                Tn = int(PEn * outBuf_opNum)
            
                PEkPEn_TkTn_inoutOP_list.append([[Tk, Tn], [PEk, PEn], [inBuf_opNum, outBuf_opNum]])
        return PEkPEn_TkTn_inoutOP_list



    def DLA(self, Tile_Point, M, K, N):
        if self.ROWBUFMAX_ENABLEED == False:
            print("Warning!! RowBufSize Maximum disabled!")

        [[Tk, Tn], [PEk, PEn], [inBuf_opNum, outBuf_opNum]] = Tile_Point
        RowBufSize = self.Bank_Row_size
        TileDLA_point_list = []

        # Rectify (Tk, Tn) for (K, N)
        Tk_rect = min(Tk, K)
        Tn_rect = min(Tn, N)

        k_tile_loop = Tk_rect  # Looping Tile Size
        n_tile_loop = Tn_rect  # Looping Tile Size
        k_tile_Vec = Tk_rect  # Vector Tile Size

        k_subtile_MAC = math.ceil(Tk_rect / PEk)  # SubTile Size per PE
        n_subtile_MAC = math.ceil(Tn_rect / PEn)  # SubTile Size per PE

        # original case
        k_subtile_inBank = k_subtile_MAC # SubTile saved in bank size per PE
        n_subtile_inBank = n_subtile_MAC # SubTile saved in bank size per PE
        ResNum_subtile = int(n_subtile_MAC * self.ResNumPerUnit) # SubTile-GEMV result per PE

        TileDLA_point = [k_tile_loop, n_tile_loop, k_tile_Vec, k_subtile_MAC, n_subtile_MAC, k_subtile_inBank, n_subtile_inBank, ResNum_subtile, Tk, Tn, PEk, PEn]
        TileDLA_point_list.append(TileDLA_point)

        # Tk_rect * VECbits < RowBufSize, should reshape to fully utilize RowFbus
        k_subtile_MAC_bits = int(k_subtile_MAC * self.VECbits) # seleve VECbits for Attention compatibility
        if self.ROWBUFMAX_ENABLEED and (k_subtile_MAC_bits < RowBufSize):
            RowBuf_Tk_ratio = math.floor(RowBufSize / k_subtile_MAC_bits)
            k_subtile_inBank = int(k_subtile_MAC * RowBuf_Tk_ratio)
            n_subtile_inBank = math.ceil(n_subtile_MAC / RowBuf_Tk_ratio)
            ResNum_subtile = int(n_subtile_MAC * self.ResNumPerUnit)

            TileDLA_point = [k_tile_loop, n_tile_loop, k_tile_Vec, k_subtile_MAC, n_subtile_MAC, k_subtile_inBank, n_subtile_inBank, ResNum_subtile, Tk, Tn, PEk, PEn]
            TileDLA_point_list.append(TileDLA_point)


        return TileDLA_point_list

    def Legality_check(self, M, K, N, TileDLA_point):
        [k_tile_loop, n_tile_loop, k_tile_Vec, k_subtile_MAC, n_subtile_MAC, k_subtile_inBank, n_subtile_inBank, ResNum_subtile, Tk, Tn, PEk, PEn] = TileDLA_point

        debug_str = ("##############################################\n"
                     + "k_tile_loop: {:5d}\tk_tile_Vec: {:5d}\tk_subtile_MAC: {:5d}\tk_subtile_inBank: {:5d}\tTk: {}\tPEk: {}\n".format(k_tile_loop, k_tile_Vec, k_subtile_MAC, k_subtile_inBank, Tk, PEk)
                     + "n_tile_loop: {:5d}\tn_subtile_MAC: {:5d}\tn_subtile_inBank: {:5d}\tResNum_subtile: {:5d}\tTn: {}\tPEn: {}\n".format(n_tile_loop, n_subtile_MAC, n_subtile_inBank, ResNum_subtile, Tn, PEn)
                     + "##############################################\n")

        for item in TileDLA_point:
            assert (type(item) is int and item > 0), "item is not uint, \n{}".format(debug_str)

        assert (k_tile_loop <= K and n_tile_loop <= N), 'K_N_loop is larger than K_N, \n{}'.format(debug_str)

        k_subtile_MAC_bits = k_subtile_MAC * self.VECbits
        assert k_subtile_MAC_bits <= self.localbuf_input, "inBufSize is not large enough!; inBufSize is {}, k_subtile_MAC_bits is {}".format(self.localbuf_input, k_subtile_MAC_bits)

        ResNum_subtile_bits = ResNum_subtile * self.VECbits
        assert ResNum_subtile_bits <= self.localbuf_output, "outBufSize is not large enough!; outBufSize is {}, ResNum_subtile_bits is {}".format(self.localbuf_output, ResNum_subtile_bits)

        assert PEk * PEn <= self.PE_num, "PEk PEn is larger!, \n{}".format(debug_str)


        assert k_tile_loop == k_tile_Vec, "k_tile_loop != k_tile_Vec!, \n{}".format(debug_str)

        VECtile_Loop_amount = k_tile_loop * self.VECbits
        MATtile_Loop_amount = k_tile_loop * n_tile_loop * self.MATbits

        VECtile_MAC_amount = k_subtile_MAC * self.VECbits * PEk
        MATtile_MAC_amount = k_subtile_MAC * n_subtile_MAC * self.MATbits * PEk * PEn

        MATtile_inBank_amount = k_subtile_inBank * n_subtile_inBank * self.MATbits * PEk * PEn

        assert VECtile_MAC_amount >= VECtile_Loop_amount, "Less MAC VECtile data!, \n{}".format(debug_str)
        assert MATtile_MAC_amount >= MATtile_Loop_amount, "Less MAC MATtile data!, \n{}".format(debug_str)
        assert MATtile_inBank_amount >= MATtile_Loop_amount, "Less inBank MATtile data!, \n{}".format(debug_str)





    def PIMCube_ExecLat_eval(self, M, K, N, SW_param):
        [loop_seq, TileDLA_point] = SW_param
        [k_tile_loop, n_tile_loop, k_tile_Vec, k_subtile_MAC, n_subtile_MAC, k_subtile_inBank, n_subtile_inBank, ResNum_subtile, Tk, Tn, PEk, PEn] = TileDLA_point

        ##### Legality Checking
        self.Legality_check(M=M, K=K, N=N, TileDLA_point=TileDLA_point)

        ##### Repreasent DataType's effect on Tile Size
        MAC_lat_extra = math.ceil(n_subtile_inBank / math.floor(self.VECbits / self.MATbits))

        ##### Loop for Complete Tiles
        latency_total = 0
        latency_VecTotal = 0
        latency_MACTotal = 0
        latency_ResTotal = 0

        ceil_K_KTileLoop = math.ceil(K / k_tile_loop)
        ceil_N_NTileLoop = math.ceil(N / n_tile_loop)
        # Latency Eval
        if loop_seq == 'mnk':
            if ceil_K_KTileLoop > 1:
                if USE_LOOP:
                    for m_idx in range(M):
                        for n_idx in range(math.ceil(N / n_tile_loop)):
                            for k_idx in range(math.ceil(K / k_tile_loop)):
                                latency_VECtile = self.activate_VECtile(k_tile=k_tile_Vec)  # k_idx
                                latency_VecTotal = latency_VecTotal + latency_VECtile
                                latency_MACtile = self.MAC_subtile(k_subtile=k_subtile_inBank) * MAC_lat_extra  # k_idx, n_idx
                                latency_MACTotal = latency_MACTotal + latency_MACtile # \todo{Shold consider N % (n_tile * n_tile_extra) here!!!}
                            latency_result = self.result_save_subtile(ResNum_subtile=ResNum_subtile)  # \todo{Shold consider N % (n_tile * n_tile_extra) here!!!}
                            latency_ResTotal = latency_ResTotal + latency_result  # n_idx
                else:
                    latency_VECtile = self.activate_VECtile(k_tile=k_tile_Vec)
                    latency_VecTotal = latency_VECtile * math.ceil(K / k_tile_loop) * math.ceil(N / n_tile_loop) * M

                    latency_MACtile = self.MAC_subtile(k_subtile=k_subtile_inBank)
                    latency_MACTotal = latency_MACtile * MAC_lat_extra * math.ceil(K / k_tile_loop) * math.ceil(N / n_tile_loop) * M

                    latency_result = self.result_save_subtile(ResNum_subtile=ResNum_subtile)
                    latency_ResTotal = latency_result * math.ceil(N / n_tile_loop) * M


            elif ceil_K_KTileLoop == 1:
                if USE_LOOP:
                    for m_idx in range(M):
                        latency_VECtile = self.activate_VECtile(k_tile=k_tile_Vec)
                        latency_VecTotal = latency_VecTotal + latency_VECtile
                        for n_idx in range(math.ceil(N / n_tile_loop)):
                            latency_MACtile = self.MAC_subtile(k_subtile=k_subtile_inBank) * MAC_lat_extra
                            latency_MACTotal = latency_MACTotal + latency_MACtile # \todo{Shold consider N % (n_tile * n_tile_extra) here!!!}
                            latency_result = self.result_save_subtile(ResNum_subtile=ResNum_subtile)  # \todo{Shold consider N % (n_tile * n_tile_extra) here!!!}
                            latency_ResTotal = latency_ResTotal + latency_result
                else:
                    latency_VECtile = self.activate_VECtile(k_tile=k_tile_Vec)
                    latency_VecTotal = latency_VECtile * M

                    latency_MACtile = self.MAC_subtile(k_subtile=k_subtile_inBank)
                    latency_MACTotal = latency_MACtile * MAC_lat_extra * math.ceil(N / n_tile_loop) * M

                    latency_result = self.result_save_subtile(ResNum_subtile=ResNum_subtile)
                    latency_ResTotal = latency_result * math.ceil(N / n_tile_loop) * M

        elif loop_seq == 'mkn':
            if ceil_N_NTileLoop > 1:
                if USE_LOOP:
                    for m_idx in range(M):
                        for k_idx in range(math.ceil(K / k_tile_loop)):
                            latency_VECtile = self.activate_VECtile(k_tile=k_tile_Vec)  # k_idx
                            latency_VecTotal = latency_VecTotal + latency_VECtile
                            for n_idx in range(math.ceil(N / n_tile_loop)):
                                latency_MACtile = self.MAC_subtile(k_subtile=k_subtile_inBank) * MAC_lat_extra  # k_idx, n_idx
                                latency_MACTotal = latency_MACTotal + latency_MACtile # \todo{Shold consider N % (n_tile * n_tile_extra) here!!!}
                                latency_result = self.result_save_subtile(ResNum_subtile=ResNum_subtile)  # \todo{Shold consider N % (n_tile * n_tile_extra) here!!!}
                                latency_ResTotal = latency_ResTotal + latency_result  # n_idx
                else:
                    latency_VECtile = self.activate_VECtile(k_tile=k_tile_Vec)
                    latency_VecTotal = latency_VECtile * math.ceil(K / k_tile_loop) * M

                    latency_MACtile = self.MAC_subtile(k_subtile=k_subtile_inBank)
                    latency_MACTotal = latency_MACtile * MAC_lat_extra * math.ceil(K / k_tile_loop) * math.ceil(N / n_tile_loop) * M

                    latency_result = self.result_save_subtile(ResNum_subtile=ResNum_subtile)
                    latency_ResTotal = latency_result * math.ceil(K / k_tile_loop) * math.ceil(N / n_tile_loop) * M

            elif ceil_N_NTileLoop == 1:
                if USE_LOOP:
                    for m_idx in range(M):
                        for k_idx in range(math.ceil(K / k_tile_loop)):
                            latency_VECtile = self.activate_VECtile(k_tile=k_tile_Vec)
                            latency_VecTotal = latency_VecTotal + latency_VECtile
                            latency_MACtile = self.MAC_subtile(k_subtile=k_subtile_inBank) * MAC_lat_extra
                            latency_MACTotal = latency_MACTotal + latency_MACtile # \todo{Shold consider N % (n_tile * n_tile_extra) here!!!}
                        latency_result = self.result_save_subtile(ResNum_subtile=ResNum_subtile)  # \todo{Shold consider N % (n_tile * n_tile_extra) here!!!}
                        latency_ResTotal = latency_ResTotal + latency_result
                else:
                    latency_VECtile = self.activate_VECtile(k_tile=k_tile_Vec)
                    latency_VecTotal = latency_VECtile * math.ceil(K / k_tile_loop) * M

                    latency_MACtile = self.MAC_subtile(k_subtile=k_subtile_inBank)
                    latency_MACTotal = latency_MACtile * MAC_lat_extra * math.ceil(K / k_tile_loop) * M

                    latency_result = self.result_save_subtile(ResNum_subtile=ResNum_subtile)
                    latency_ResTotal = latency_result * M

        else:
            print("Wrong loop_seq @\"{}\"".format(loop_seq))
            exit()

        # Summarize total time
        if self.RESSAVE_INCLUDED:
            latency_total = latency_VecTotal + latency_MACTotal + latency_ResTotal
        else:
            print("WARNING: Latency_ResTotal Excluded")
            latency_total = latency_VecTotal + latency_MACTotal


        # MAC_FLOPs Eval
        MACflops_once = 2 * k_subtile_inBank * n_subtile_MAC * MAC_lat_extra * self.PE_num
        MAC_FLOPs_total = MACflops_once * math.ceil(K / k_tile_loop) * math.ceil(N / n_tile_loop) * M


        # To calculate OI
        FLOPS_total = (K + K) * N * M
        # BYTES_Vector = int(K * M * math.ceil(N / n_tile_loop) * self.VECbits / 8)
        # BYTES_MAC = int(N * K * M * self.MATbits / 8)
        # BYTES_Result = int(N * M * self.VECbits / 8)
        # BYTES_total = BYTES_Vector + BYTES_MAC + BYTES_Result
        # OI = FLOPS_total / BYTES_total
        BYTES_Vector = 0
        BYTES_MAC = 0
        BYTES_Result = 0
        BYTES_total = 0
        OI = 0

        cycles_total = int(latency_total / self.tCK)
        cycles_VEC = int(latency_VecTotal / self.tCK)
        cycles_MAC = int(latency_MACTotal / self.tCK)
        cycles_RES = int(latency_ResTotal / self.tCK)

        lat_total = cycles_total / self.CLK_freq / 1e3

        MAC_Utilze = FLOPS_total / MAC_FLOPs_total

        result_list = [[cycles_total, lat_total],
                       [cycles_VEC, cycles_MAC, cycles_RES, BYTES_Vector, BYTES_MAC, BYTES_Result, FLOPS_total, MAC_FLOPs_total, MAC_Utilze],
                       [k_tile_loop, n_tile_loop, k_tile_Vec, k_subtile_MAC, n_subtile_MAC, k_subtile_inBank, n_subtile_inBank, ResNum_subtile, Tk, Tn, PEk, PEn],
                       ]
        return result_list

    def SyncTransLat_eval(self, xPU_BW, DataOP_amount, OPBits):
        # only considering data transfering time, since they are always slower than data saving/loadingto/from DRAM
        DRAM_BW = self.pChannel_num * self.DQ_bitwidth * self.CLK_freq / 8 # Bytes/s #todo{tod verift the CLKFreq}
        DataPath_BW = min(xPU_BW, DRAM_BW)
        DataTrans_btis = DataOP_amount * OPBits
        TransLat_perCH = DataTrans_btis / DataPath_BW * 1e3 # in ms
        return TransLat_perCH



    def xPU_PIM_Trans_eval(self, Mapping_param, now_PIM_PEn):
        [model_batch, model_dhead, model_hdim, last_layer, layer, last_mapping, now_mapping, last_PIM_PEk, xPU_BW, OPBits] = Mapping_param

        # Determin Data_amount
        numOP_PerPAX = math.ceil(layer.numOp / self.PAX_Num)
        ## last QKVgen, now score
        if last_layer.name in ['qkv']:
            M_tx = last_layer.m
            N_tx = math.ceil(last_layer.n / self.PAX_Num)
            Data_amount_tx = M_tx * N_tx * last_PIM_PEk  ## PIM -> xPU (reduce and append KVcache)
            M_rx = 1
            N_rx = model_dhead
            Data_amount_rx = M_rx * N_rx * numOP_PerPAX  ## xPU -> PIM (scatter)
        ## last score, now context
        elif last_layer.name in ['score']:
            M_tx = last_layer.m
            N_tx = last_layer.n
            Data_amount_tx = M_tx * N_tx * last_PIM_PEk * numOP_PerPAX  ## PIM -> xPU (reduce)
            M_rx = M_tx
            N_rx = N_tx
            Data_amount_rx = M_rx * N_rx * numOP_PerPAX  ## xPU -> PIM (scatter)
        ## last context, now Proj
        elif last_layer.name in ['context']:
            M_tx = last_layer.m
            N_tx = last_layer.n
            Data_amount_tx = M_tx * N_tx * last_PIM_PEk * numOP_PerPAX  ## PIM -> xPU (reduce and concate and re-batch)
            M_rx = model_batch
            N_rx = model_hdim
            Data_amount_rx = M_rx * N_rx * now_PIM_PEn  ## xPU -> PIM (broadcast)
        elif last_layer.name in ['proj', 'ff1', 'ff2', 'ff3']:
            M_tx = last_layer.m
            N_tx = math.ceil(last_layer.n / self.PAX_Num)
            Data_amount_tx = M_tx * N_tx * last_PIM_PEk  ## PIM -> xPU (reduce and concate)
            M_rx = last_layer.m
            N_rx = last_layer.n
            Data_amount_rx = M_rx * N_rx * now_PIM_PEn  ## xPU -> PIM (broadcast)
        else:
            print("WRONG last_layer_name!")
            exit()

        assert last_mapping in ['PIM', 'xPU'], "Wrong last_maping"
        assert now_mapping in ['PIM', 'xPU'], "Wrong last_maping"
        if last_mapping == 'PIM':
            assert last_PIM_PEk != None, 'Last on PIM while UNknow last_PIM_PEk'
            if now_mapping == 'xPU': # PIM -> xPU
                assert now_PIM_PEn == None, 'on xPU while UNknow now_PIM_PEn'
                Trans_lat_tx = self.SyncTransLat_eval(xPU_BW=xPU_BW, DataOP_amount=Data_amount_tx, OPBits=OPBits)
                Trans_lat = Trans_lat_tx
            else: # PIM -> xPU -> PIM
                assert now_PIM_PEn != None, 'on PIM while UNknow now_PIM_PEn'
                Trans_lat_tx = self.SyncTransLat_eval(xPU_BW=xPU_BW, DataOP_amount=Data_amount_tx, OPBits=OPBits)
                Trans_lat_rx = self.SyncTransLat_eval(xPU_BW=xPU_BW, DataOP_amount=Data_amount_rx, OPBits=OPBits)
                Trans_lat = Trans_lat_tx + Trans_lat_rx
        else:
            assert last_PIM_PEk == None, 'Last on xPU while UNknow last_PIM_PEk'
            if now_mapping == 'xPU': # xPU -> xPU
                assert now_PIM_PEn == None, 'on xPU while UNknow now_PIM_PEn'
                Trans_lat = 0
            else: # xPU -> PIM
                assert now_PIM_PEn != None, 'on PIM while UNknow now_PIM_PEn'
                Trans_lat_rx = self.SyncTransLat_eval(xPU_BW=xPU_BW, DataOP_amount=Data_amount_rx, OPBits=OPBits)
                Trans_lat = Trans_lat_rx

        return Trans_lat


    def find_minCycs(self, M, K, N, VECbits, MATbits, SIMDlane_num, Mapping_param, noTrans=False):
        ########### inital parameters relevant to VECbits and MATBits
        self.VECbits = VECbits
        self.MATbits = MATbits
        assert self.MATbits <= self.VECbits, "MATbits larger than VECbits!!"

        self.SIMDlane_num = SIMDlane_num
        self.MAC_Bitwidth = math.floor(self.VECbits / self.MATbits) * self.MATbits
        self.MAC_Engine_size = self.MAC_Bitwidth * self.SIMDlane_num
        
        self.SubTile_k_MAX = math.floor(self.localbuf_input / self.VECbits)
        self.SubTile_n_MAX = math.floor(self.localbuf_output / self.VECbits)


        ########### Run Baseline ############
        if self.Run_Baseline:

            # print("\tWARNING!!!!!\tthe FLOW was FORCED to {}-PIM".format(self.PIM_type))
            if self.PIM_type == 'HBM2':
                PEk = 1
                PEn = 128
                # PEn = 32
                inBuf_opNum = 128
                outBuf_opNum = 8
                Tk = int(PEk * inBuf_opNum)
                Tn = int(PEn * outBuf_opNum)
                Tile_Point = [[Tk, Tn], [PEk, PEn], [inBuf_opNum, outBuf_opNum]]
                self.RESSAVE_INCLUDED = True
                self.ROWBUFMAX_ENABLEED = True

                # [k_tile_loop, n_tile_loop, k_tile_Vec, k_subtile_MAC, n_subtile_MAC, k_subtile_inBank, n_subtile_inBank, Tk, Tn, PEk, PEn]
                TileDLA_point_list = self.DLA(Tile_Point=Tile_Point, M=M, K=K, N=N)
                TileDLA_point = TileDLA_point_list[1]
                SW_param = ['mnk', TileDLA_point]
            elif self.PIM_type == 'LPDDR4':
                PEk = 1
                PEn = 64
                inBuf_opNum = 2048
                outBuf_opNum = 1
                Tk = int(PEk * inBuf_opNum)
                Tn = int(PEn * outBuf_opNum)
                Tile_Point = [[Tk, Tn], [PEk, PEn], [inBuf_opNum, outBuf_opNum]]
                self.RESSAVE_INCLUDED = False
                self.ROWBUFMAX_ENABLEED = False

                # [k_tile_loop, n_tile_loop, k_tile_Vec, k_subtile_MAC, n_subtile_MAC, k_subtile_inBank, n_subtile_inBank, Tk, Tn, PEk, PEn]
                TileDLA_point_list = self.DLA(Tile_Point=Tile_Point, M=M, K=K, N=N)
                TileDLA_point = TileDLA_point_list[0]
                SW_param = ['mnk', TileDLA_point]
            else:
                print('WRONG PIMtype {}'.format(self.PIM_type))
                exit()

            SWparam_DS = [SW_param]
            total_num = 1
        ########### Profiling Tile and LNO Design Space ############
        else:
            TileCase_list = self.Build_Tile_DS()
            tile_case_num = len(TileCase_list)

            LNOCase_list = ['mnk', 'mkn']
            LNO_case_num = len(LNOCase_list)

            SWparam_DS = []
            for i in range(LNO_case_num):
                LNO_Point = LNOCase_list[i]
                for j in range(tile_case_num):
                    Tile_Point = TileCase_list[j]
                    TileDLA_point_list = self.DLA(Tile_Point=Tile_Point, M=M, K=K, N=N)
                    for TileDLA_point in TileDLA_point_list:
                        SWparam_DS.append([LNO_Point, TileDLA_point])

            total_num = len(SWparam_DS)


        ### Applying DSE and find optimal
        result = np.zeros([total_num, 23])
        for idx in range(total_num):
            ############## Eval PIM Computation Latency
            SW_param = SWparam_DS[idx]

            [PIMtime_res, partial_res, debug_res] = self.PIMCube_ExecLat_eval(M=M, K=K, N=N, SW_param=SW_param)
            [PIM_Exec_cycles, PIM_comp_lat] = PIMtime_res # num 2
            [cycles_VEC, cycles_MAC, cycles_RES, BYTES_Vector, BYTES_MAC, BYTES_Result, FLOPS_total, MAC_FLOPs_total, MAC_Utilze] = partial_res # num 9
            [k_tile_loop, n_tile_loop, k_tile_Vec, k_subtile_MAC, n_subtile_MAC, k_subtile_inBank, n_subtile_inBank, ResNum_subtile, Tk, Tn, PEk, PEn] = debug_res # num 12

            ############## Eval Transaction Latency
            if not noTrans:
                PIM_Trans_lat = self.xPU_PIM_Trans_eval(Mapping_param=Mapping_param, now_PIM_PEn=PEn)
            else:
                PIM_Trans_lat = 0
            PIM_exec_lat = PIM_comp_lat + PIM_Trans_lat

            ############## return result
            time_res = [PIM_exec_lat, PIM_comp_lat, PIM_Trans_lat]
            result[idx, :] = np.expand_dims(np.array(time_res + partial_res + debug_res), axis=0)
        min_idx = np.argmin(result[:, 0])
        best_case = result[min_idx, :]

        Execlat_min = best_case[0]
        Complat_min = best_case[1]
        Translat_min = best_case[2]

        return [Execlat_min, Complat_min, Translat_min, best_case]# result]