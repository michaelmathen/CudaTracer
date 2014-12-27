	.text
	.globl _main
_main:
LFB77:
	movss	LC0(%rip), %xmm0
	movaps	%xmm0, %xmm1
	mulss	LC1(%rip), %xmm1
	addss	%xmm1, %xmm0
	cvttss2si	%xmm0, %eax
	ret
LFE77:
	.literal4
	.align 2
LC0:
	.long	1085066445
	.align 2
LC1:
	.long	0
	.section __TEXT,__eh_frame,coalesced,no_toc+strip_static_syms+live_support
EH_frame1:
	.set L$set$0,LECIE1-LSCIE1
	.long L$set$0
LSCIE1:
	.long	0
	.byte	0x1
	.ascii "zR\0"
	.byte	0x1
	.byte	0x78
	.byte	0x10
	.byte	0x1
	.byte	0x10
	.byte	0xc
	.byte	0x7
	.byte	0x8
	.byte	0x90
	.byte	0x1
	.align 3
LECIE1:
LSFDE1:
	.set L$set$1,LEFDE1-LASFDE1
	.long L$set$1
LASFDE1:
	.long	LASFDE1-EH_frame1
	.quad	LFB77-.
	.set L$set$2,LFE77-LFB77
	.quad L$set$2
	.byte	0
	.align 3
LEFDE1:
	.constructor
	.destructor
	.align 1
	.subsections_via_symbols
