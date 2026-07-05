classdef Shape_To_PadLayer1000 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        CNN1_Split_split
        CNN1_Pad_value
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end


    methods(Static, Hidden)
        % Specify the path to the class that will be used for codegen
        function name = matlabCodegenRedirect(~)
            name = 'modclass_vbx.coder.Shape_To_PadLayer1000';
        end
    end


    methods
        function this = Shape_To_PadLayer1000(name)
            this.Name = name;
            this.OutputNames = {'CNN1_Pad'};
        end

        function [CNN1_Pad] = predict(this, InputLayer)
            if isdlarray(InputLayer)
                InputLayer = stripdims(InputLayer);
            end
            InputLayerNumDims = 4;
            InputLayer = modclass_vbx.ops.permuteInputVar(InputLayer, [4 3 1 2], 4);

            [CNN1_Pad, CNN1_PadNumDims] = Shape_To_PadGraph1000(this, InputLayer, InputLayerNumDims, false);
            CNN1_Pad = modclass_vbx.ops.permuteOutputVar(CNN1_Pad, [3 4 2 1], 4);

            CNN1_Pad = dlarray(single(CNN1_Pad), 'SSCB');
        end

        function [CNN1_Pad] = forward(this, InputLayer)
            if isdlarray(InputLayer)
                InputLayer = stripdims(InputLayer);
            end
            InputLayerNumDims = 4;
            InputLayer = modclass_vbx.ops.permuteInputVar(InputLayer, [4 3 1 2], 4);

            [CNN1_Pad, CNN1_PadNumDims] = Shape_To_PadGraph1000(this, InputLayer, InputLayerNumDims, true);
            CNN1_Pad = modclass_vbx.ops.permuteOutputVar(CNN1_Pad, [3 4 2 1], 4);

            CNN1_Pad = dlarray(single(CNN1_Pad), 'SSCB');
        end

        function [CNN1_Pad, CNN1_PadNumDims1001] = Shape_To_PadGraph1000(this, InputLayer, InputLayerNumDims, Training)

            % Execute the operators:
            % Shape:
            [CNN1_InputShapeInt, CNN1_InputShapeIntNumDims] = modclass_vbx.ops.onnxShape(InputLayer, InputLayerNumDims, 0, InputLayerNumDims+1);

            % Cast:
            CNN1_InputShape = single(CNN1_InputShapeInt);
            CNN1_InputShapeNumDims = CNN1_InputShapeIntNumDims;

            % Div:
            CNN1_Div = CNN1_InputShape ./ this.Vars.CNN1_Stride;
            CNN1_DivNumDims = max(CNN1_InputShapeNumDims, this.NumDims.CNN1_Stride);

            % Ceil:
            CNN1_Ceil = ceil(CNN1_Div);
            CNN1_CeilNumDims = CNN1_DivNumDims;

            % Sub:
            CNN1_Sub = CNN1_Ceil - this.Vars.CNN1_One;
            CNN1_SubNumDims = max(CNN1_CeilNumDims, this.NumDims.CNN1_One);

            % Mul:
            CNN1_Mul = CNN1_Sub .* this.Vars.CNN1_Stride;
            CNN1_MulNumDims = max(CNN1_SubNumDims, this.NumDims.CNN1_Stride);

            % Add:
            CNN1_Add = CNN1_Mul + this.Vars.CNN1_Filter;
            CNN1_AddNumDims = max(CNN1_MulNumDims, this.NumDims.CNN1_Filter);

            % Sub:
            CNN1_Sub1 = CNN1_Add - CNN1_InputShape;
            CNN1_Sub1NumDims = max(CNN1_AddNumDims, CNN1_InputShapeNumDims);

            % Relu:
            CNN1_ReluInt = relu(dlarray(CNN1_Sub1));
            CNN1_ReluIntNumDims = CNN1_Sub1NumDims;

            % Cast:
            CNN1_Relu = single(CNN1_ReluInt);
            CNN1_ReluNumDims = CNN1_ReluIntNumDims;

            % Split:
            [CNN1_BCPadding, CNN1_totalPaddingNee, CNN1_totalPaddingN_5, CNN1_BCPaddingNumDims, CNN1_totalPaddingNeeNumDims, CNN1_totalPaddingN_5NumDims] = modclass_vbx.ops.onnxSplit13(CNN1_Relu, 0, this.CNN1_Split_split, 3, CNN1_ReluNumDims);

            % Div:
            CNN1_totalPaddingN_2 = CNN1_totalPaddingNee ./ this.Vars.CNN1_totalPaddingN_4;
            CNN1_totalPaddingN_2NumDims = max(CNN1_totalPaddingNeeNumDims, this.NumDims.CNN1_totalPaddingN_4);

            % Floor:
            CNN1_totalPaddingN_3 = floor(CNN1_totalPaddingN_2);
            CNN1_totalPaddingN_3NumDims = CNN1_totalPaddingN_2NumDims;

            % Ceil:
            CNN1_totalPaddingN_1 = ceil(CNN1_totalPaddingN_2);
            CNN1_totalPaddingN_1NumDims = CNN1_totalPaddingN_2NumDims;

            % Div:
            CNN1_totalPaddingN_7 = CNN1_totalPaddingN_5 ./ this.Vars.CNN1_totalPaddingN_9;
            CNN1_totalPaddingN_7NumDims = max(CNN1_totalPaddingN_5NumDims, this.NumDims.CNN1_totalPaddingN_9);

            % Floor:
            CNN1_totalPaddingN_8 = floor(CNN1_totalPaddingN_7);
            CNN1_totalPaddingN_8NumDims = CNN1_totalPaddingN_7NumDims;

            % Ceil:
            CNN1_totalPaddingN_6 = ceil(CNN1_totalPaddingN_7);
            CNN1_totalPaddingN_6NumDims = CNN1_totalPaddingN_7NumDims;

            % Concat:
            [CNN1_paddingSize, CNN1_paddingSizeNumDims] = modclass_vbx.ops.onnxConcat(0, {this.Vars.CNN1_Zero, this.Vars.CNN1_Zero, CNN1_totalPaddingN_3, CNN1_totalPaddingN_8, this.Vars.CNN1_Zero, this.Vars.CNN1_Zero, CNN1_totalPaddingN_1, CNN1_totalPaddingN_6}, [this.NumDims.CNN1_Zero, this.NumDims.CNN1_Zero, CNN1_totalPaddingN_3NumDims, CNN1_totalPaddingN_8NumDims, this.NumDims.CNN1_Zero, this.NumDims.CNN1_Zero, CNN1_totalPaddingN_1NumDims, CNN1_totalPaddingN_6NumDims]);

            % Cast:
            CNN1_paddingSizeInt = cast(int64(extractdata(CNN1_paddingSize)), 'like', CNN1_paddingSize);
            CNN1_paddingSizeIntNumDims = CNN1_paddingSizeNumDims;

            % Pad:
            [CNN1_Pad, CNN1_PadNumDims] = modclass_vbx.ops.onnxPad(InputLayer, CNN1_paddingSizeInt, this.CNN1_Pad_value, 'constant', dlarray([0:InputLayerNumDims]'), InputLayerNumDims);

            % Set graph output arguments
            CNN1_PadNumDims1001 = CNN1_PadNumDims;

        end

    end

end