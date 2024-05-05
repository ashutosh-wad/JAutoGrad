package com.ashutoshwad.utils.jautograd.exception;

public class JAutogradException extends RuntimeException {
	private static final long serialVersionUID = 1L;

	public JAutogradException() {
		super();
	}

	public JAutogradException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
		super(message, cause, enableSuppression, writableStackTrace);
	}

	public JAutogradException(String message, Throwable cause) {
		super(message, cause);
	}

	public JAutogradException(String message) {
		super(message);
	}

	public JAutogradException(Throwable cause) {
		super(cause);
	}

}
