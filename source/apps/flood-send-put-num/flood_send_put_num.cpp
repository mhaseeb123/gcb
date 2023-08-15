
#include "flood_send_put_num.hpp"

//#define MAX_MSG_SIZE 16 //doubles
#define MAX_MSG_SIZE 4096 //doubles
#define RES_ROW 9
#define RES_COL 9

/* message size in bytes  */
/* 0 = 1byte (latency) */
/* 3 = 8bytes (hashtable, one-sided) */
/* 5 = 32byte (hashtable two-sided ~24bytes) */
/* 16 = 65536byte (stencil) */
int msg_size[9]={0,3,5,8,11,13,16,18,20};


template <typename T>
__global__ void fill_buff(T *buff, int size)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int el = id; el < size; el += blockDim.x * gridDim.x)
        buff[el] = (T) (el + 0.1) / (el + 1);
}

template<typename T>
__global__ void fill_buff(T *buff, int rank, int size, T val)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int el = id; el < size; el += blockDim.x * gridDim.x)
        buff[el] = (T) (val);
}

template<typename T>
__global__ void fill_buff(T *buff, int size, int rank)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int el = id; el < size; el += blockDim.x * gridDim.x)
        buff[el] = (T) ((rank + 0.1) / (el + 1));
}

#define MPI_CHECK(stmt)                                                                         \
    do {                                                                                        \
        int result = (stmt);                                                                    \
        if (MPI_SUCCESS != result) {                                                            \
            fprintf(stderr, "[%s:%d] MPI failed with error %d \n", __FILE__, __LINE__, result); \
            exit(-1);                                                                           \
        }                                                                                       \
    } while (0)

#define         CRC_POLY_16             0xA001
#define         CRC_START_16            0x0000
#define         CRC_START_8             0x00
int             crc_tab16_init          = 0;
uint16_t         crc_tab16[256];
uint8_t sht75_crc_table[256] = {

        0,   49,  98,  83,  196, 245, 166, 151, 185, 136, 219, 234, 125, 76,  31,  46,
        67,  114, 33,  16,  135, 182, 229, 212, 250, 203, 152, 169, 62,  15,  92,  109,
        134, 183, 228, 213, 66,  115, 32,  17,  63,  14,  93,  108, 251, 202, 153, 168,
        197, 244, 167, 150, 1,   48,  99,  82,  124, 77,  30,  47,  184, 137, 218, 235,
        61,  12,  95,  110, 249, 200, 155, 170, 132, 181, 230, 215, 64,  113, 34,  19,
        126, 79,  28,  45,  186, 139, 216, 233, 199, 246, 165, 148, 3,   50,  97,  80,
        187, 138, 217, 232, 127, 78,  29,  44,  2,   51,  96,  81,  198, 247, 164, 149,
        248, 201, 154, 171, 60,  13,  94,  111, 65,  112, 35,  18,  133, 180, 231, 214,
        122, 75,  24,  41,  190, 143, 220, 237, 195, 242, 161, 144, 7,   54,  101, 84,
        57,  8,   91,  106, 253, 204, 159, 174, 128, 177, 226, 211, 68,  117, 38,  23,
        252, 205, 158, 175, 56,  9,   90,  107, 69,  116, 39,  22,  129, 176, 227, 210,
        191, 142, 221, 236, 123, 74,  25,  40,  6,   55,  100, 85,  194, 243, 160, 145,
        71,  118, 37,  20,  131, 178, 225, 208, 254, 207, 156, 173, 58,  11,  88,  105,
        4,   53,  102, 87,  192, 241, 162, 147, 189, 140, 223, 238, 121, 72,  27,  42,
        193, 240, 163, 146, 5,   52,  103, 86,  120, 73,  26,  43,  188, 141, 222, 239,
        130, 179, 224, 209, 70,  119, 36,  21,  59,  10,  89,  104, 255, 206, 157, 172
};
/*
 * static void init_crc16_tab( void );
 *
 * For optimal performance uses the CRC16 routine a lookup table with values
 * that can be used directly in the XOR arithmetic in the algorithm. This
 * lookup table is calculated by the init_crc16_tab() routine, the first time
 * the CRC function is called.
 */

void init_crc16_tab( void ) {

    uint16_t i;
    uint16_t j;
    uint16_t crc;
    uint16_t c;

    for (i=0; i<256; i++) {

        crc = 0;
        c   = i;

        for (j=0; j<8; j++) {

            if ( (crc ^ c) & 0x0001 ) crc = ( crc >> 1 ) ^ CRC_POLY_16;
            else                      crc =   crc >> 1;

            c = c >> 1;
        }

        crc_tab16[i] = crc;
    }

    crc_tab16_init = 1;

}  /* init_crc16_tab */

/*
 * uint16_t crc_16( const unsigned char *input_str, size_t num_bytes );
 *
 * The function crc_16() calculates the 16 bits CRC16 in one pass for a byte
 * string of which the beginning has been passed to the function. The number of
 * bytes to check is also a parameter. The number of the bytes in the string is
 * limited by the constant SIZE_MAX.
 */
uint16_t crc_16( const unsigned char *input_str, size_t num_bytes ) {

    uint16_t crc;
    const unsigned char *ptr;
    size_t a;

    if ( ! crc_tab16_init ) init_crc16_tab();

    crc = CRC_START_16;
    ptr = input_str;

    if ( ptr != NULL ) for (a=0; a<num_bytes; a++) {

            crc = (crc >> 8) ^ crc_tab16[ (crc ^ (uint16_t) *ptr++) & 0x00FF ];
        }

    return crc;

}  /* crc_16 */

uint8_t crc_8( const unsigned char *input_str, size_t num_bytes ) {

    size_t a;
    uint8_t crc;
    const unsigned char *ptr;

    crc = CRC_START_8;
    ptr = input_str;

    if ( ptr != NULL ) for (a=0; a<num_bytes; a++) {

            crc = sht75_crc_table[(*ptr++) ^ crc];
        }

    return crc;

}  /* crc_8 */

/* Borrowed from util-linux-2.13-pre7/schedutils/taskset.c */
static char *cpuset_to_cstr(cpu_set_t *mask, char *str)
{
    char *ptr = str;
    int i, j, entry_made = 0;
    for (i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, mask)) {
            int run = 0;
            entry_made = 1;
            for (j = i + 1; j < CPU_SETSIZE; j++) {
                if (CPU_ISSET(j, mask)) run++;
                else break;
            }
            if (!run)
                sprintf(ptr, "%d,", i);
            else if (run == 1) {
                sprintf(ptr, "%d,%d,", i, i + 1);
                i++;
            } else {
                sprintf(ptr, "%d-%d,", i, i + run);
                i += run;
            }
            while (*ptr != 0) ptr++;
        }
    }
    ptr -= entry_made;
    *ptr = 0;
    return(str);
}

void print_output(int num_elem, double* res, double* res_lat, int nprocs,int rank, int peer_skip){
    if(rank==0){
        printf("size/rank,\t");
        fflush(stdout);
        for (int peer=1; peer<nprocs; peer=peer+peer_skip){
            printf("%-8d,\t",peer);
            fflush(stdout);
        }
        printf("\n");
        fflush(stdout);
        int res_idx=0;
        for(int mysize=1; mysize<=num_elem;mysize=mysize*2){
            printf("%-8lu,\t", mysize*sizeof(double));
            fflush(stdout);
            for (int peer=1; peer<nprocs; peer=peer+peer_skip){
                printf(" %-8.2f,\t %-8.2f,", res[res_idx*(nprocs)+peer],res_lat[res_idx*(nprocs)+peer]);
                fflush(stdout);
            }
            printf("\n");
            fflush(stdout);
            res_idx+=1;
        }
    }
    memset(res, 0, (MAX_MSG_SIZE) * (nprocs+1)* sizeof(double));
}

//
// two sided test
//
void two_sided_test(int iter, int skip, int nprocs, int rank,  int peer_skip,int flight_num_id, int start_peer, int start_size_id){
    double t_comm=0;

    if (rank==0) {
        printf("Two-sided Isend/Recv, skip=%d, iter=%d, peer_skip=%d/%d\n",skip,iter,peer_skip,nprocs);
        fflush(stdout);
    }

    MPI_Request send_req ;
    MPI_Status recv_status ;
    for (int peer = start_peer; peer < nprocs; peer=peer+peer_skip) {
        for (int id = start_size_id; id < RES_ROW; id++) {
            int mysize = (int) pow(2, msg_size[id]);
            MPI_Datatype send_type;
            MPI_Type_contiguous(mysize, MPI_CHAR, &send_type);
            MPI_Type_commit(&send_type);
            if (rank == 0) printf("Commit MPI_Type, bytes=%d(%d), sizeof(char)=%lu\n", mysize, id, sizeof(char));
            fflush(stdout);

            char *s_buf, *r_buf;
            int s_buf_size = (mysize + 1);
            cudaMalloc(&s_buf, s_buf_size * sizeof(char));
            cudaMalloc(&r_buf, s_buf_size * sizeof(char));

            if (rank == 0) printf("---- malloc %f MB\n", (s_buf_size * sizeof(char)) / 1e6);
            fflush(stdout);

            constexpr int bsize = 256;
            int gsize = s_buf_size/bsize + (s_buf_size%bsize==0?0:1);

            fill_buff<<<gsize,bsize>>>(s_buf, s_buf_size);
            fill_buff<<<gsize, bsize>>>(r_buf, s_buf_size, (char)0);

            cudaDeviceSynchronize();
/*
            for (int ii = 0; ii < s_buf_size; ii++) {
                s_buf[ii] = (ii + 0.1) / (ii + 1);
            }

            for (int jj = 0; jj < s_buf_size; jj++) {
                r_buf[jj] = 0.00;
            }
*/
            for (int f = flight_num_id; f < RES_COL; f++) {
                int msg_sync = pow(10, f);
                int myiter, myskip;
                //if (f==3){
                //    myiter=10000;
                //} else if (f==4){
                //    myiter=1000;
                //}else if (f==5){
                //    myiter=100;
                //}else if (f==6){
                //    myiter=10;
                //}else{
                //    myiter=iter;
                //}
                myiter=iter;
                while (myiter * msg_sync >= 1e6) {
                    myiter = 0.1*myiter;
                }
                if (f==6) myiter=1;
                myskip=myiter*0.1;
                if (rank == 0) printf("---- message per sync %d. iter=%d + %d\n", msg_sync, myiter, myskip);
                fflush(stdout);

                t_comm = 0.0;
                MPI_Barrier(MPI_COMM_WORLD);
                for (int i = 0; i < myiter + myskip; i++) {
                    if (i >= myskip) t_comm += -MPI_Wtime();
                    if (rank == 0) {
                        for (int j = 0; j < msg_sync; j++) {
                            MPI_Isend(&s_buf[0], 1, send_type, peer, j, MPI_COMM_WORLD, &send_req);
                        }
                    } else if (rank == peer) {
                        for (int j = 0; j < msg_sync; j++) {
                            MPI_Recv(&r_buf[0], 1, send_type, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_status);
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (i >= myskip) t_comm += MPI_Wtime();
                } //iter

                if (rank == 0) {
                    //latency = (latency * 1e6) / iter; // us
                    //bw = (mysize * sizeof(double)) / (latency); // B/us == MB/s
                    //res[res_idx * (nprocs) + peer] = bw;
                    //res_lat[res_idx * (nprocs) + peer] = latency;
                    //res_idx += 1;
                    printf("size=%d, msg_sync=%d, peer=%d, time per sync=%.6f us, bw=%.2f MB/s, iter=%d, skip=%d\n", mysize, msg_sync,
                           peer, (t_comm * 1e6) / myiter, (mysize*msg_sync) / ((t_comm * 1e6) / myiter), myiter, myskip);
                    fflush(stdout);
                }

            } // msg_sync
            MPI_Type_free(&send_type);
            cudaFree(s_buf);
            cudaFree(r_buf);
        } // msg size
    } // peer
} // END TWO_SIDED

//
// put data flush theoretical
//
void put_data_flush_theo(int iter, int skip, int nprocs, int rank,  int peer_skip, int flight_num_id, int start_peer, int start_size_id) {
double t_comm=0;
    if (rank==0) {
        printf("one-sided put(data+1), skip=%d, iter=%d, peer_skip=%d/%d\n",skip,iter,peer_skip,nprocs);
        fflush(stdout);
    }

   for (int peer = start_peer; peer < nprocs; peer=peer+peer_skip) {
        for (int id = start_size_id; id < RES_ROW; id++) {
            int mysize = (int) pow(2, msg_size[id]);
            MPI_Datatype send_type;
            MPI_Type_contiguous(mysize+1, MPI_CHAR, &send_type);
            MPI_Type_commit(&send_type);
            if (rank == 0) printf("Commit MPI_Type, bytes=%d(%d), sizeof(char)=%lu\n", mysize+1, id, sizeof(char));
            fflush(stdout);

            for (int f = flight_num_id; f < RES_COL; f++) {
                int msg_sync = pow(10, f);
                int myiter, myskip;
                if (f==4){
                    myiter=10000;
                }else if (f==5){
                    myiter=1000;
                }else if (f==6){
                    myiter=100;
                }else{
                    myiter=iter;
                }
                myskip=myiter*0.1;
                //if (rank == 0) printf("---- message per sync %d\n", msg_sync);
                //fflush(stdout);
                int s_buf_size = (mysize + 1);
                char *s_buf;
                cudaMalloc(&s_buf, s_buf_size * sizeof(char));
                MPI_Win my_winl;
                MPI_Win_create(s_buf,s_buf_size * sizeof(char), sizeof(char), MPI_INFO_NULL, MPI_COMM_WORLD, &my_winl);
                //if (rank == 0) printf("---- malloc %f MB\n", (s_buf_size * sizeof(char)) / 1e6);
                //fflush(stdout);

                constexpr int bsize = 256;
                int gsize = s_buf_size/bsize + (s_buf_size%bsize==0?0:1);

                fill_buff<<<gsize, bsize>>>(s_buf, s_buf_size, rank);
                cudaDeviceSynchronize();

                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Win_lock_all(0, my_winl);
                t_comm = 0.0;
                for (int i = 0; i < myiter + myskip; i++) {
                    if (i >= myskip) t_comm += -MPI_Wtime();
                    if (rank == 0) {
                        for (int j = 0; j < msg_sync; j++) {
                            MPI_Put(&s_buf[0], 1, send_type, peer, 0, 1, send_type, my_winl);
                            MPI_Win_flush(peer, my_winl);
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (i >= myskip) t_comm += MPI_Wtime();
                } //iter
                MPI_Win_unlock_all(my_winl);
                if (rank == 0) {
                    //latency = (latency * 1e6) / iter; // us
                    //bw = (mysize * sizeof(double)) / (latency); // B/us == MB/s
                    //res[res_idx * (nprocs) + peer] = bw;
                    //res_lat[res_idx * (nprocs) + peer] = latency;
                    //res_idx += 1;
                    printf("size=%d, msg_sync=%d, peer=%d, time per sync=%.6f us, bw=%.2f MB/s, iter=%d, skip=%d\n", mysize+1, msg_sync,
                           peer, (t_comm * 1e6) / myiter, ((mysize+1)*msg_sync) / ((t_comm * 1e6) / myiter), myiter, myskip);
                    fflush(stdout);
                }
                cudaFree(s_buf);
            } // msg_sync
            MPI_Type_free(&send_type);
        } // msg size
    } // peer

} //END put_data_flush_theo

//
// put data flush atomic sum
//
void put_data_flush_atomic_sum(int iter, int skip, int nprocs, int rank,  int peer_skip,int flight_num_id, int start_peer, int start_size_id) {
    double t_comm=0;
    if (rank==0) {
        printf("one-sided fetch and op sum, skip=%d, iter=%d, peer_skip=%d/%d\n",skip,iter,peer_skip,nprocs);
        fflush(stdout);
    }

    for (int peer = start_peer; peer < nprocs; peer=peer+peer_skip) {
        for (int id = RES_ROW-1; id < RES_ROW; id++) {
            int s_buf_size = 1;
            int64_t *s_buf;

            MPI_Win my_winl;
            //if (id==0) {
            //    send_type = MPI_INT8_T;
            //    MPI_Win_allocate(s_buf_size*sizeof(int64_t), sizeof(int8_t), MPI_INFO_NULL, MPI_COMM_WORLD, &s_buf, &my_winl);
            //    if (rank == 0) printf(" MPI_Type=%d,sizeof(int8_t)=%lu\n", send_type, sizeof(int8_t));
            //}else if (id==1) {
            //    send_type = MPI_INT16_T;
            //    MPI_Win_allocate(s_buf_size*sizeof(int64_t), sizeof(int16_t), MPI_INFO_NULL, MPI_COMM_WORLD, &s_buf, &my_winl);
            //    if (rank == 0) printf(" MPI_Type=%d,sizeof(int16_t)=%lu\n", send_type, sizeof(int16_t));
            //}else if (id==2) {
            //    send_type = MPI_INT32_T;
            //    MPI_Win_allocate(s_buf_size*sizeof(int64_t), sizeof(int32_t), MPI_INFO_NULL, MPI_COMM_WORLD, &s_buf, &my_winl);
            //    if (rank == 0) printf(" MPI_Type=%d,sizeof(int32_t)=%lu\n", send_type, sizeof(int32_t));
            //}else if (id==3) {
            //    send_type = MPI_INT64_T;
            MPI_Win_allocate(s_buf_size*sizeof(int64_t), sizeof(int64_t), MPI_INFO_NULL, MPI_COMM_WORLD, &s_buf, &my_winl);
            //    if (rank == 0) printf(" MPI_Type=%d,sizeof(int64_t)=%lu\n", send_type, sizeof(int64_t));
            //}else{
            //    break;
            //}


            for (int f = flight_num_id; f < RES_COL; f++) {
                int msg_sync = pow(10, f);
                int myiter, myskip;
                if (f==4){
                    myiter=100;
                }else if (f==5){
                    myiter=10;
                }else if (f==6){
                    myiter=1;
                }else{
                    myiter=iter;
                }
                myskip=myiter*0.1;
                if (rank == 0) printf("---- message per sync %d\n", msg_sync);
                fflush(stdout);

                //constexpr int bsize = 256;
                //int gsize = s_buf_size/bsize + (s_buf_size%bsize==0?0:1);
                //fill_buff<<<gsize, bsize>>>(s_buf, s_buf_size, (int64_t)(-1));

                cudaDeviceSynchronize();

                for (int ii = 0; ii < s_buf_size; ii++) {
                    s_buf[ii] = -1;
                }

                int64_t *res;
                int64_t *elem;

                cudaMalloc(&res, sizeof(int64_t));
                cudaMalloc(&elem, sizeof(int64_t));

                int64_t h_res=0,h_elem=1;

                cudaMemcpy(res, &h_res, sizeof(int64_t), cudaMemcpyHostToDevice);
                cudaMemcpy(elem, &h_elem, sizeof(int64_t), cudaMemcpyHostToDevice);

                cudaDeviceSynchronize();

                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Win_lock_all(0, my_winl);
                t_comm = 0.0;
                for (int i = 0; i < myiter + myskip; i++) {
                    if (i >= myskip) t_comm = t_comm -MPI_Wtime();
                    if (rank == 0) {
                        for (int j = 0; j < msg_sync; j++) {
                            //MPI_Put(&s_buf[0], 1, send_type, peer, 0, 1, send_type, my_winl);
                            MPI_Fetch_and_op(/* origin */ elem, /* result */ res, MPI_INT64_T, peer, /* disp */ 0, MPI_SUM,my_winl );
                            //MPI_Compare_and_swap(elem, compare, res, send_type, peer, /* target disp */ 0, my_winl);
                            MPI_Win_flush(peer, my_winl);
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (i >= myskip) t_comm += MPI_Wtime();
                } //iter
                MPI_Win_unlock_all(my_winl);
                if (rank == 0) {
                    //latency = (latency * 1e6) / iter; // us
                    //bw = (mysize * sizeof(double)) / (latency); // B/us == MB/s
                    //res[res_idx * (nprocs) + peer] = bw;
                    //res_lat[res_idx * (nprocs) + peer] = latency;
                    //res_idx += 1;
                    printf("size=%lu, msg_sync=%d, peer=%d, time per sync=%.6f us,iter=%d, skip=%d\n", sizeof(int64_t), msg_sync,
                           peer, (t_comm * 1e6) / myiter,  myiter,myskip);

                    fflush(stdout);
                }
            } // msg_sync
        } // msg size
    } // peer
} // END put_data_flush_atomic_sum

//
// put data flush atomic
//
void put_data_flush_atomic(int iter, int skip, int nprocs, int rank,  int peer_skip,int flight_num_id, int start_peer, int start_size_id) {
    double t_comm=0;
    if (rank==0) {
        printf("one-sided compare and swap, skip=%d, iter=%d, peer_skip=%d/%d\n",skip,iter,peer_skip,nprocs);
        fflush(stdout);
    }

   for (int peer = start_peer; peer < nprocs; peer=peer+peer_skip) {
        for (int id =RES_ROW-1 ; id < RES_ROW; id++) {

            int s_buf_size = 1;
            int64_t *s_buf;
            MPI_Win my_winl;

            //if (id==0) {
            //    send_type = MPI_INT8_T;
            //    MPI_Win_allocate(s_buf_size*sizeof(int64_t), sizeof(int8_t), MPI_INFO_NULL, MPI_COMM_WORLD, &s_buf, &my_winl);
            //}else if (id==1) {
            //    send_type = MPI_INT16_T;
            //    MPI_Win_allocate(s_buf_size*sizeof(int64_t), sizeof(int16_t), MPI_INFO_NULL, MPI_COMM_WORLD, &s_buf, &my_winl);
            //}else if (id==2) {
            //    send_type = MPI_INT32_T;
            //    MPI_Win_allocate(s_buf_size*sizeof(int64_t), sizeof(int32_t), MPI_INFO_NULL, MPI_COMM_WORLD, &s_buf, &my_winl);
            //}else if (id==3) {
                MPI_Win_allocate(s_buf_size*sizeof(int64_t), sizeof(MPI_INT64_T), MPI_INFO_NULL, MPI_COMM_WORLD, &s_buf, &my_winl);
            //}else{
            //    break;
            //}


            for (int f = flight_num_id; f < RES_COL; f++) {
                int msg_sync = pow(10, f);
                int myiter, myskip;
                if (f==4){
                    myiter=100;
                }else if (f==5){
                    myiter=10;
                }else if (f==6){
                    myiter=1;
                }else{
                    myiter=iter;
                }
                myskip=myiter*0.1;
                if (rank == 0) printf("---- message per sync %d\n", msg_sync);
                fflush(stdout);

                for (int ii = 0; ii < s_buf_size; ii++) {
                    s_buf[ii] = 0;
                }

                int64_t h_compare=0,h_res=-1,h_elem=0;
                int64_t *compare,*res,*elem;

                cudaMalloc(&compare, sizeof(int64_t));
                cudaMalloc(&res, sizeof(int64_t));
                cudaMalloc(&elem, sizeof(int64_t));

                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Win_lock_all(0, my_winl);
                t_comm = 0.0;
                for (int i = 0; i < myiter + myskip; i++) {
                    double t_gpu = 0.0;
                    if (i >= myskip) t_comm = t_comm-MPI_Wtime();
                    if (rank == 0) {
                        for (int j = 0; j < msg_sync; j++) {
                            double t1 = MPI_Wtime();
                            h_compare=j,h_res=-1,h_elem=j+1;
                            cudaMemcpy(compare, &h_compare, sizeof(int64_t), cudaMemcpyHostToDevice);
                            cudaMemcpy(res, &h_res, sizeof(int64_t), cudaMemcpyHostToDevice);
                            cudaMemcpy(elem, &h_elem, sizeof(int64_t), cudaMemcpyHostToDevice);
                            t_gpu += (MPI_Wtime() - t1);
                            //MPI_Put(&s_buf[0], 1, send_type, peer, 0, 1, send_type, my_winl);
                            MPI_Compare_and_swap(elem, compare, res, MPI_INT64_T, peer, /* target disp */ 0, my_winl);
                            MPI_Win_flush(peer, my_winl);
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (i >= myskip) t_comm += MPI_Wtime() - t_gpu;
                } //iter
                MPI_Win_unlock_all(my_winl);
                if (rank == 0) {
                    //latency = (latency * 1e6) / iter; // us
                    //bw = (mysize * sizeof(double)) / (latency); // B/us == MB/s
                    //res[res_idx * (nprocs) + peer] = bw;
                    //res_lat[res_idx * (nprocs) + peer] = latency;
                    //res_idx += 1;
                    printf("size=%lu, msg_sync=%d, peer=%d, time per sync=%.6f us,iter=%d, skip=%d\n", sizeof(int64_t),msg_sync,
                           peer, (t_comm * 1e6) / myiter, myiter,myskip);

                    fflush(stdout);
                }
            } // msg_sync
        } // msg size
    } // peer
} // END put_data_flush_atomic


//
// put data flush op
//
void put_data_flush_op(int iter, int skip, int nprocs, int rank,  int peer_skip,int flight_num_id, int start_peer, int start_size_id) {
    double t_comm=0;
    if (rank==0) {
        printf("one-sided put(data+1), skip=%d, iter=%d,  peer_skip=%d/%d\n",skip,iter,peer_skip,nprocs);
        fflush(stdout);
    }

   for (int peer = start_peer; peer < nprocs; peer=peer+peer_skip) {
        for (int id = start_size_id; id < RES_ROW; id++) {
            int mysize = (int) pow(2, msg_size[id]);
            MPI_Datatype send_type;
            MPI_Type_contiguous(mysize, MPI_CHAR, &send_type);
            MPI_Type_commit(&send_type);
            if (rank == 0)
                printf("Commit MPI_Type, bytes=%d(%d), sizeof(char)=%lu\n", mysize, id, sizeof(char));
            fflush(stdout);

            for (int f = flight_num_id; f < RES_COL; f++) {
                int msg_sync = pow(10, f);
                int myiter, myskip;
                if (f==4){
                    myiter=10000;
                }else if (f==5){
                    myiter=1000;
                }else if (f==6){
                    myiter=100;
                }else{
                    myiter=iter;
                }
                myskip=myiter*0.1;
                if (rank == 0) printf("---- message per sync %d\n", msg_sync);
                fflush(stdout);
                int s_buf_size = (mysize + 1);
                char *s_buf;
                cudaMalloc(&s_buf, s_buf_size * sizeof(char));
                MPI_Win my_winl;
                MPI_Win_create(s_buf,s_buf_size * sizeof(char), sizeof(char), MPI_INFO_NULL, MPI_COMM_WORLD, &my_winl);
                if (rank == 0) printf("---- malloc %f MB\n", (s_buf_size * sizeof(char)) / 1e6);
                fflush(stdout);

                constexpr int bsize = 256;
                int gsize = s_buf_size/bsize + (s_buf_size%bsize==0?0:1);

                fill_buff<<<gsize, bsize>>>(s_buf, s_buf_size);
                cudaDeviceSynchronize();

                /*
                for (int ii = 0; ii < s_buf_size; ii++) {
                    s_buf[ii] = (rank + 0.1) / (ii + 1);
                }
                */

                MPI_Barrier(MPI_COMM_WORLD);
                MPI_Win_lock_all(0, my_winl);
                t_comm = 0.0;
                for (int i = 0; i < myiter + myskip; i++) {
                    if (i >= myskip) t_comm += -MPI_Wtime();
                    if (rank == 0) {
                        for (int j = 0; j < msg_sync; j++) {
                            MPI_Put(&s_buf[0], 1, send_type, peer, 0, 1, send_type, my_winl);
                            MPI_Win_flush_local(peer, my_winl);
                            MPI_Put(&s_buf[mysize], 1, MPI_CHAR, peer, mysize, 1, MPI_CHAR, my_winl);
                            MPI_Win_flush(peer, my_winl);
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                    if (i >= myskip) t_comm += MPI_Wtime();
                } //iter
                MPI_Win_unlock_all(my_winl);
                if (rank == 0) {
                    //latency = (latency * 1e6) / iter; // us
                    //bw = (mysize * sizeof(double)) / (latency); // B/us == MB/s
                    //res[res_idx * (nprocs) + peer] = bw;
                    //res_lat[res_idx * (nprocs) + peer] = latency;
                    //res_idx += 1;
                    printf("size=%d, msg_sync=%d, peer=%d, time per sync=%.6f us, bw=%.2f MB/s, iter=%d, skip=%d\n", mysize, msg_sync,
                           peer, (t_comm * 1e6) / myiter, (msg_sync * mysize) / ((t_comm * 1e6) / myiter), myiter,myskip);
                    fflush(stdout);
                }
                cudaFree(s_buf);
            } // msg_sync
            MPI_Type_free(&send_type);
        } // msg size
    } // peer
} // END put_data_flush_op

//
// main function
//
int main(int c, char *v[]) {
    int iam, nprocs;

    MPI_CHECK(MPI_Init(&c, &v));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &iam));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));

#define check_mpi_tag
#ifdef check_mpi_tag
    void *value;
    int  tag_ub, isSet;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &value, &isSet);

    tag_ub = *(int *) value;

    if (iam==0){
    if (isSet)
        printf("rank %d: attribute MPI_TAG_UB for MPI_COMM_WORLD is %d\n",iam, tag_ub);
    else
        printf("rank %d: attribute MPI_TAG_UB for MPI_COMM_WORLD is NOT set\n",iam);
    fflush(stdout);
    }
#endif
    int iter=atoi(v[1]);
    int flight_num_id=atoi(v[2]);
    int peer_skip=atoi(v[3]);
    int start_peer=atoi(v[4]);
    int start_size_id=atoi(v[5]);

    int skip=0.1*iter;
    if (iter<100) skip=0;

    if (iam > 0)
    {
        cudaSetDevice(1+(iam%3));
    }

#ifdef cpu_bind
    cpu_set_t my_set;        /* Define your cpu_set bit mask. */
    CPU_ZERO(&my_set);       /* Initialize it all to 0, i.e. no CPUs selected. */
    CPU_SET(iam*2, &my_set);     /* set the bit that represents core 7. */
    sched_setaffinity(0, sizeof(cpu_set_t), &my_set); /* Set affinity of tihs process to */
                                                    /* the defined mask, i.e. only 7. */

    //cpu_set_t coremask;
    //char clbuf[7 * CPU_SETSIZE], hnbuf[64];

    //memset(clbuf, 0, sizeof(clbuf));
    //memset(hnbuf, 0, sizeof(hnbuf));
    //(void)gethostname(hnbuf, sizeof(hnbuf));
    //(void)sched_getaffinity(0, sizeof(coremask), &coremask);
    //cpuset_to_cstr(&coremask, clbuf);
    //int cpu = sched_getcpu();
    //int node = numa_node_of_cpu(cpu);
    //printf("Hello from rank %d, CPU %d (core affinity = %s), NUMA node %d, on %s,\n", iam, cpu, clbuf, node,hnbuf);
    //fflush(stdout);

#endif

    MPI_Barrier(MPI_COMM_WORLD);

//#define two_sided
#ifdef two_sided
    two_sided_test(iter,skip,nprocs,iam,peer_skip, flight_num_id, start_peer,start_size_id);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

// #define one_sided_flush_atomic
#ifdef one_sided_flush_atomic
    put_data_flush_atomic(iter,skip,nprocs,iam,peer_skip,flight_num_id, start_peer,start_size_id);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

#define one_sided_flush_atomic_sum
#ifdef one_sided_flush_atomic_sum
    put_data_flush_atomic_sum(iter,skip,nprocs,iam,peer_skip,flight_num_id, start_peer,start_size_id);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

//#define one_sided_flush_theo
#ifdef one_sided_flush_theo
    put_data_flush_theo(iter,skip,nprocs,iam,peer_skip,flight_num_id, start_peer,start_size_id);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

//#define one_sided_flush_data_op
#ifdef one_sided_flush_data_op
    put_data_flush_op(iter,skip,nprocs,iam,peer_skip,flight_num_id, start_peer,start_size_id);
    MPI_Barrier(MPI_COMM_WORLD);
#endif

//#ifdef one_sided_crc
//    put_data_crc16(iter,skip,nprocs,iam,peer_skip,flight_num);
//    MPI_Barrier(MPI_COMM_WORLD);
//#endif
//#ifdef one_sided_lock
//    put_data_unlockall(iter,skip,nprocs,iam,peer_skip,flight_num);
//    MPI_Barrier(MPI_COMM_WORLD);
//#endif

    MPI_Finalize();
}
