classdef Shape_To_PadLayer1001 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        CNN2_Split_split
        CNN2_Pad_value
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
            name = 'modclass_vbx.coder.Shape_To_PadLayer1001';
        end
    end


    methods
        function this = Shape_To_PadLayer1001(name)
            this.Name = name;
            this.OutputNames = {'CNN2_Pad'};
        end

        function [CNN2_Pad] = predict(this, MaxPool1)
            if isdlarray(MaxPool1)
                MaxPool1 = stripdims(MaxPool1);
            end
            MaxPool1NumDims = 4;
            MaxPool1 = modclass_vbx.ops.permuteInputVar(MaxPool1, [4 3 1 2], 4);

            [CNN2_Pad, CNN2_PadNumDims] = Shape_To_PadGraph1002(this, MaxPool1, MaxPool1NumDims, false);
            CNN2_Pad = modclass_vbx.ops.permuteOutputVar(CNN2_Pad, [3 4 2 1], 4);

            CNN2_Pad = dlarray(single(CNN2_Pad), 'SSCB');
        end

        function [CNN2_Pad] = forward(this, MaxPool1)
            if isdlarray(MaxPool1)
                MaxPool1 = stripdims(MaxPool1);
            end
            MaxPool1NumDims = 4;
            MaxPool1 = modclass_vbx.ops.permuteInputVar(MaxPool1, [4 3 1 2], 4);

            [CNN2_Pad, CNN2_PadNumDims] = Shape_To_PadGraph1002(this, MaxPool1, MaxPool1NumDims, true);
            CNN2_Pad = modclass_vbx.ops.permuteOutputVar(CNN2_Pad, [3 4 2 1], 4);

            CNN2_Pad = dlarray(single(CNN2_Pad), 'SSCB');
        end

        function [CNN2_Pad, CNN2_PadNumDims1003] = Shape_To_PadGraph1002(this, MaxPool1, MaxPool1NumDims, Training)

            % Execute the operators:
            % Shape:
            [CNN2_InputShapeInt, CNN2_InputShapeIntNumDims] = modclass_vbx.ops.onnxShape(MaxPool1, MaxPool1NumDims, 0, MaxPool1NumDims+1);

            % Cast:
            CNN2_InputShape = single(CNN2_InputShapeInt);
            CNN2_InputShapeNumDims = CNN2_InputShapeIntNumDims;

            % Div:
            CNN2_Div = CNN2_InputShape ./ this.Vars.CNN2_Stride;
            CNN2_DivNumDims = max(CNN2_InputShapeNumDims, this.NumDims.CNN2_Stride);

            % Ceil:
            CNN2_Ceil = ceil(CNN2_Div);
            CNN2_CeilNumDims = CNN2_DivNumDims;

            % Sub:
            CNN2_Sub = CNN2_Ceil - this.Vars.CNN2_One;
            CNN2_SubNumDims = max(CNN2_CeilNumDims, this.NumDims.CNN2_One);

            % Mul:
            CNN2_Mul = CNN2_Sub .* this.Vars.CNN2_Stride;
            CNN2_MulNumDims = max(CNN2_SubNumDims, this.NumDims.CNN2_Stride);

            % Add:
            CNN2_Add = CNN2_Mul + this.Vars.CNN2_Filter;
            CNN2_AddNumDims = max(CNN2_MulNumDims, this.NumDims.CNN2_Filter);

            % Sub:
            CNN2_Sub1 = CNN2_Add - CNN2_InputShape;
            CNN2_Sub1NumDims = max(CNN2_AddNumDims, CNN2_InputShapeNumDims);

            % Relu:
            CNN2_ReluInt = relu(dlarray(CNN2_Sub1));
            CNN2_ReluIntNumDims = CNN2_Sub1NumDims;

            % Cast:
            CNN2_Relu = single(CNN2_ReluInt);
            CNN2_ReluNumDims = CNN2_ReluIntNumDims;

            % Split:
            [CNN2_BCPadding, CNN2_totalPaddingNee, CNN2_totalPaddingN_5, CNN2_BCPaddingNumDims, CNN2_totalPaddingNeeNumDims, CNN2_totalPaddingN_5NumDims] = modclass_vbx.ops.onnxSplit13(CNN2_Relu, 0, this.CNN2_Split_split, 3, CNN2_ReluNumDims);

            % Div:
            CNN2_totalPaddingN_2 = CNN2_totalPaddingNee ./ this.Vars.CNN2_totalPaddingN_4;
            CNN2_totalPaddingN_2NumDims = max(CNN2_totalPaddingNeeNumDims, this.NumDims.CNN2_totalPaddingN_4);

            % Floor:
            CNN2_totalPaddingN_3 = floor(CNN2_totalPaddingN_2);
            CNN2_totalPaddingN_3NumDims = CNN2_totalPaddingN_2NumDims;

            % Ceil:
            CNN2_totalPaddingN_1 = ceil(CNN2_totalPaddingN_2);
            CNN2_totalPaddingN_1NumDims = CNN2_totalPaddingN_2NumDims;

            % Div:
            CNN2_totalPaddingN_7 = CNN2_totalPaddingN_5 ./ this.Vars.CNN2_totalPaddingN_9;
            CNN2_totalPaddingN_7NumDims = max(CNN2_totalPaddingN_5NumDims, this.NumDims.CNN2_totalPaddingN_9);

            % Floor:
            CNN2_totalPaddingN_8 = floor(CNN2_totalPaddingN_7);
            CNN2_totalPaddingN_8NumDims = CNN2_totalPaddingN_7NumDims;

            % Ceil:
            CNN2_totalPaddingN_6 = ceil(CNN2_totalPaddingN_7);
            CNN2_totalPaddingN_6NumDims = CNN2_totalPaddingN_7NumDims;

            % Concat:
            [CNN2_paddingSize, CNN2_paddingSizeNumDims] = modclass_vbx.ops.onnxConcat(0, {this.Vars.CNN2_Zero, this.Vars.CNN2_Zero, CNN2_totalPaddingN_3, CNN2_totalPaddingN_8, this.Vars.CNN2_Zero, this.Vars.CNN2_Zero, CNN2_totalPaddingN_1, CNN2_totalPaddingN_6}, [this.NumDims.CNN2_Zero, this.NumDims.CNN2_Zero, CNN2_totalPaddingN_3NumDims, CNN2_totalPaddingN_8NumDims, this.NumDims.CNN2_Zero, this.NumDims.CNN2_Zero, CNN2_totalPaddingN_1NumDims, CNN2_totalPaddingN_6NumDims]);

            % Cast:
            CNN2_paddingSizeInt = cast(int64(extractdata(CNN2_paddingSize)), 'like', CNN2_paddingSize);
            CNN2_paddingSizeIntNumDims = CNN2_paddingSizeNumDims;

            % Pad:
            [CNN2_Pad, CNN2_PadNumDims] = modclass_vbx.ops.onnxPad(MaxPool1, CNN2_paddingSizeInt, this.CNN2_Pad_value, 'constant', dlarray([0:MaxPool1NumDims]'), MaxPool1NumDims);

            % Set graph output arguments
            CNN2_PadNumDims1003 = CNN2_PadNumDims;

        end

    end

end