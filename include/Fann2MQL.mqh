/* Fann2MQL.mqh
 *
 * Copyright (C) 2008-2010 Mariusz Woloszyn
 *
 *  This file is part of Fann2MQL package
 *
 *  Fann2MQL is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  Fann2MQL is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with Fann2MQL; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#property copyright "Mariusz Woloszyn"
#property link ""

#import "Fann2MQL.dll"

#define F2M_MAX_THREADS 64

/* Creation/Execution */
int f2M_create_standard(int num_layers, int l1num, int l2num, int l3num, int l4num);
int f2M_create_standard_array(int num_layers, unsigned int &layers[]);
int f2M_destroy(int ann);
int f2M_destroy_all_anns();
int f2M_run(int ann, double& input_vector[]);
double f2M_get_output(int ann, int output);
int f2M_randomize_weights(int ann, double min_weight, double max_weight);
/* Creation/Execution Parameters */
int f2M_get_num_input(int ann);
int f2M_get_num_output(int ann);
int f2M_get_total_neurons(int ann);
int f2M_get_num_layers(int ann);

/* Training */
int f2M_train(int ann, double& input_vector[], double& output_vector[]);
int f2M_train_fast(int ann, double& input_vector[], double& output_vector[]);
int f2M_test(int ann, double& input_vector[], double& output_vector[]);
double f2M_get_MSE(int ann);
int f2M_get_bit_fail(int ann);
int f2M_reset_MSE(int ann);
/* Training Parameters */
int f2M_get_training_algorithm(int ann);
int f2M_set_training_algorithm(int ann, int training_algorithm);
int f2M_set_act_function_layer(int ann, int activation_function, int layer);
int f2M_set_act_function_hidden(int ann, int activation_function);
int f2M_set_act_function_output(int ann, int activation_function);

/* Data training */
int f2M_train_on_file(int ann, char& filename[], int max_epoch, double desired_error);

/* File Input/Output */
int f2M_create_from_file(char& path[]);
int f2M_save(int ann, char& path[]);

/* Parallel processing functions */
int f2M_parallel_init();
int f2M_parallel_deinit();
int f2M_run_parallel(int anns_count, int& anns[], double& input_vector[]);
int f2M_train_parallel(int anns_count, int& anns[], double& input_vector[], double& output_vector[]);
#import

int f2M_train_on_file(int ann, string filename, int max_epoch, double desired_error)
{
	uchar p[];

	StringToCharArray(filename, p, 0, -1, CP_ACP);

	return f2M_train_on_file(ann, p, max_epoch, desired_error);
}

int f2M_create_from_file(string path)
{
	uchar p[];

	StringToCharArray(path, p, 0, -1, CP_ACP);

	return f2M_create_from_file(p);
}

int f2M_save(int ann, string path)
{
	uchar p[];

	StringToCharArray(path, p, 0, -1, CP_ACP);

	return f2M_save(ann, p);
}

enum fann_activationfunc_enum
{
	FANN_LINEAR = 0,
	FANN_THRESHOLD,
	FANN_THRESHOLD_SYMMETRIC,
	FANN_SIGMOID,
	FANN_SIGMOID_STEPWISE,
	FANN_SIGMOID_SYMMETRIC,
	FANN_SIGMOID_SYMMETRIC_STEPWISE,
	FANN_GAUSSIAN,
	FANN_GAUSSIAN_SYMMETRIC,
	// Stepwise linear approximation to gaussian.
	// Faster than gaussian but a bit less precise.
	// NOT implemented yet.
	FANN_GAUSSIAN_STEPWISE,
	FANN_ELLIOT,
	FANN_ELLIOT_SYMMETRIC,
	FANN_LINEAR_PIECE,
	FANN_LINEAR_PIECE_SYMMETRIC,
	FANN_SIN_SYMMETRIC,
	FANN_COS_SYMMETRIC,
	FANN_SIN,
	FANN_COS
};

static string FANN_ACTIVATIONFUNC_NAMES[] = {
	"FANN_LINEAR",
	"FANN_THRESHOLD",
	"FANN_THRESHOLD_SYMMETRIC",
	"FANN_SIGMOID",
	"FANN_SIGMOID_STEPWISE",
	"FANN_SIGMOID_SYMMETRIC",
	"FANN_SIGMOID_SYMMETRIC_STEPWISE",
	"FANN_GAUSSIAN",
	"FANN_GAUSSIAN_SYMMETRIC",
	"FANN_GAUSSIAN_STEPWISE",
	"FANN_ELLIOT",
	"FANN_ELLIOT_SYMMETRIC",
	"FANN_LINEAR_PIECE",
	"FANN_LINEAR_PIECE_SYMMETRIC",
	"FANN_SIN_SYMMETRIC",
	"FANN_COS_SYMMETRIC",
	"FANN_SIN",
	"FANN_COS"
};

enum fann_train_enum
{
	FANN_TRAIN_INCREMENTAL = 0,
	FANN_TRAIN_BATCH,
	FANN_TRAIN_RPROP,
	FANN_TRAIN_QUICKPROP,
	FANN_TRAIN_SARPROP
};

static string FANN_TRAIN_NAMES[] = {
	"FANN_TRAIN_INCREMENTAL",
	"FANN_TRAIN_BATCH",
	"FANN_TRAIN_RPROP",
	"FANN_TRAIN_QUICKPROP",
	"FANN_TRAIN_SARPROP"
};

enum fann_errorfunc_enum
{
        FANN_ERRORFUNC_LINEAR = 0,
	FANN_ERRORFUNC_TANH
};

static string FANN_ERRORFUNC_NAMES[] = {
        "FANN_ERRORFUNC_LINEAR",
	"FANN_ERRORFUNC_TANH"
};

enum fann_stopfunc_enum
{
        FANN_STOPFUNC_MSE = 0,
	FANN_STOPFUNC_BIT
};

static string FANN_STOPFUNC_NAMES[] = {
        "FANN_STOPFUNC_MSE",
	"FANN_STOPFUNC_BIT"
};

enum fann_nettype_enum
{
	FANN_NETTYPE_LAYER = 0, /* Each layer only has connections to the next layer */
        FANN_NETTYPE_SHORTCUT /* Each layer has connections to all following layers */
};

static string FANN_NETTYPE_NAMES[] = {
        "FANN_NETTYPE_LAYER",
	"FANN_NETTYPE_SHORTCUT"
};

#define FANN_DOUBLE_ERROR -1000000000

enum fann_errno_enum
{
        FANN_E_NO_ERROR = 0,
	FANN_E_CANT_OPEN_CONFIG_R,
	FANN_E_CANT_OPEN_CONFIG_W,
	FANN_E_WRONG_CONFIG_VERSION,
	FANN_E_CANT_READ_CONFIG,
	FANN_E_CANT_READ_NEURON,
	FANN_E_CANT_READ_CONNECTIONS,
	FANN_E_WRONG_NUM_CONNECTIONS,
	FANN_E_CANT_OPEN_TD_W,
	FANN_E_CANT_OPEN_TD_R,
	FANN_E_CANT_READ_TD,
	FANN_E_CANT_ALLOCATE_MEM,
	FANN_E_CANT_TRAIN_ACTIVATION,
	FANN_E_CANT_USE_ACTIVATION,
	FANN_E_TRAIN_DATA_MISMATCH,
	FANN_E_CANT_USE_TRAIN_ALG,
	FANN_E_TRAIN_DATA_SUBSET,
	FANN_E_INDEX_OUT_OF_BOUND,
	FANN_E_SCALE_NOT_PRESENT,
	FANN_E_INPUT_NO_MATCH,
	FANN_E_OUTPUT_NO_MATCH,
	FANN_E_WRONG_PARAMETERS_FOR_CREATE
};

struct fann_error
{
        fann_errno_enum errno_f;
	int error_log; // ErrLog FileHandle
	string errstr; // uchar errstr[]
};
